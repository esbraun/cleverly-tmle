# Implementation validation studies

Each section validates one method that `cleverly` implements. A section exists only for a study
registered in `tests/studies/evidence/registry.py`. The shared machinery in
`tests/studies/evidence/` computes every verdict, and `tests/unit/test_method_evidence.py` checks
this document against the committed results. Every table below is generated from those results, so
a stale number is a test failure and not a reading error.

The summary of all six studies is the
[implementation validation grid](index.md#implementation-validation-grid). To register a new
study, follow [adding a method row](../development/method-benchmarking.md#adding-a-method-row).

## How to read these studies

Read this section once. It defines the terms and the rules that every study below applies.

### The three questions

A study asks three separate questions about `cleverly`. The counts are not interchangeable.

| question | what it establishes about `cleverly` | what it cannot establish |
| --- | --- | --- |
| **Accuracy against known truth** | `cleverly` clears declared bias and coverage-validity margins on a law whose truth is computed longhand | nothing about the derivation or exact nominal coverage. The canonical rows say nothing about `cleverly` |
| **Agreement with the canonical implementation** | `cleverly` and an independently maintained implementation compute the same thing on the same rows, within a declared margin | that either one is right. Two poor implementations can agree. This is why the accuracy question is asked first and separately |
| **Theory properties** | declared repeated-sampling properties, including robustness, root-n contraction, calibration, and error rates | behaviour outside the laws and compositions the cells declare. Efficiency requires a separate comparison with an independent bound |

The scientific derivation is checked elsewhere. The exact-law, Gateaux, remainder, identity, and
deliberate-mutation instruments are listed per estimand in the
[evidence manifest](evidence.md#the-table). A study measures a complete estimator under repeated
sampling. It does not replace those instruments.

### The verdict rules

Every rule is an equivalence rule bounded by a margin declared before the run. None of them tests
whether a discrepancy is exactly zero, and the distinction is the design rather than a detail.

A Monte Carlo study accumulates evidence by adding replications. A verdict must therefore become
easier, or at worst stay put, as replications grow. A significance test does the reverse. It
converges on rejecting any estimator whose finite-sample remainder is not identically zero, which
is every estimator. The study would eventually go red for the one reason that is not a defect. An
earlier version of the first study carried two such rules. Quadrupling its replication count would
have failed it without a line of estimator code changing.
`tests/unit/test_evidence_framework.py` holds both rules side by side and asserts which way each
one moves.

| rule | what must hold | why it is bounded | its control |
| --- | --- | --- | --- |
| **bias equivalence** | the 99% Student interval for the error lies inside `margin:standardized_bias` empirical standard deviations of zero | a point test of zero bias fails every consistent estimator at enough replications | the same instrument read in the opposite direction: a control cell's interval must lie entirely *outside* the margin |
| **coverage validity** | the lower endpoint of the exact Clopper-Pearson interval clears `margin:coverage_floor` | whether a nominal 95% interval is valid is the question. Whether it is 95% to the third decimal is a question no finite study answers | one-sided by design. Over-coverage is conservative, not invalid |
| **SE sanity band** | the bootstrap interval for mean reported SE over empirical SD lies inside `margin:se_ratio_sanity_lower` to `margin:se_ratio_sanity_upper` | this is a screen for a standard error wrong by an order of magnitude. The coverage floor binds first | a ratio of 0.80 corresponds to about 88% coverage, so the band cannot be tightened past what the floor implies |
| **SE calibration** | where both nuisances are correct, the SE ratio interval lies inside the calibration band and the exact coverage interval inside its own | this is the only two-sided calibration claim, and the only gate a uniform tenth-scale understatement fails | a `shrunken_se_control` cell multiplies the standard errors by a declared factor and must fail |
| **efficiency** | the empirical standard deviation of the estimates and the mean reported standard error both sit inside `margin:efficiency_ratio_lower` to `margin:efficiency_ratio_upper` of an independently computed efficient-influence-function bound | Monte Carlo error in the empirical spread, which a point test against the bound would reject at enough replications. Calibration and validity are asked separately, because an estimator can have root-n bias and calibrated intervals without attaining the bound | a `noise_control` cell adds one bound-unit of independent noise and must fail |
| **paired similarity** | the 99% interval for the mean paired difference lies within `margin:paired_difference` pooled empirical standard deviations | symmetric, because a large difference in either direction means the two implementations are not computing the same thing | a paired mutation makes `cleverly` materially worse and must fail |
| **RMSE non-inferiority** | the bootstrap upper bound for `cleverly`'s RMSE ratio is at most `margin:rmse_noninferiority` | one-sided, because `cleverly` performing better than the reference is a result rather than a failure | a reference-only mutation must fail the reference's own column and leave `cleverly` standing |
| **coverage non-inferiority** | the lower bound for the coverage difference is at least `margin:coverage_noninferiority` | one-sided, for the same reason | as above |
| **calibration non-inferiority** | the upper bound for excess absolute SE-calibration error is at most `margin:calibration_noninferiority` | applied only where the two native inference scales are comparable | an exemption must be earned. A test requires the two implementations to report genuinely different scales |
| **type-I error** | the one-sided upper endpoint of the rejection rate under a confounded sharp null stays under `margin:type_i_ceiling` | the null law keeps its confounding, so the test is not an unadjusted comparison | a power cell under a real effect must reject. An inert test cannot pass by never firing |
| **power** | the rejection lower bound clears `margin:minimum_power` | this is the positive control the type-I cell needs | none. It is itself a control |
| **root-n rate** | the log-log slope interval lies within `margin:root_n_slope_lower` to `margin:root_n_slope_upper` | a band rather than containment of -1/2, which is a point test the reported-SE rate already fails at these replication counts | the interval must also exclude `margin:excluded_slope`, so a merely decreasing spread fails |

The **efficiency** rule needs a bound this package did not compute from the estimator it judges. A
study without one runs no efficiency cell, and its tables say so. The property family named
`root_n_and_efficiency` is the exception in name only. The name is historical, it tests bias,
coverage, and SE calibration across sample sizes, and the efficiency comparison lives in
`interval_calibration`.

Each implementation is judged on its own terms as well as on the pairing. A reference that degrades
is reported against the reference. It does not turn `cleverly`'s row red.

The harness has negative controls for all of it, in `tests/unit/test_method_evidence.py`. The fast
tests corrupt bias, coverage, and reported standard errors for each implementation in turn. They
require only that implementation to fail and require the untouched one to keep passing.

### Terms

| term | definition |
| --- | --- |
| **replication** | one simulated dataset drawn from a study's law, fitted by every implementation. Seeds derive from the study's own record, so replication *k* is a fixed sample whatever the study's size |
| **law** | the data-generating process a scenario samples, with a parameter computed longhand rather than estimated |
| **estimand** | the parameter a row reports. Longitudinal keys carry their regimen in brackets |
| **cell** | one configuration of a property family, labelled `positive` when it must pass the family's instrument and `control` when it must fail it |
| **margin** | an acceptance threshold declared before the run and recorded in the study's `manifest.json`. Names beginning `margin:` in a measured-values table are these thresholds |
| **standardized bias** | mean error divided by the empirical standard deviation of the estimates. Scale-free, so it reads the same across estimands |
| **SE ratio** | mean reported standard error divided by the empirical standard deviation. One means the reported uncertainty matches the real spread |
| **share of margin used** | how much of a similarity margin a paired comparison consumed. Small means the two implementations agree far inside the bound |
| **plug-in** | the estimate the same nuisance fits produce with no targeting fluctuation. The distance from it measures what targeting did |

## Canonical point-treatment TMLE

This study validates ordinary, non-cross-fitted point-treatment TMLE in `cleverly`. The canonical
comparison uses R [`tmle3`](https://github.com/tlverse/tmle3) 0.2.0 at pinned commit
[`ed72f8a`](https://github.com/tlverse/tmle3/tree/ed72f8a20e64c914ab25ffe015d865f7a9963d27). The
independent claims come from the parameter oracles and from the properties in van der Laan and
Rubin's [original TMLE paper](https://doi.org/10.2202/1557-4679.1043).

The row is deliberately named **ordinary** TMLE. The public default cross-fits its nuisances. The
comparison disables cross-fitting in `cleverly` because `tmle3`'s ordinary specs are not CV-TMLE.
Cross-fitted and CV-TMLE constructions are separate methods and do not inherit this result.

### What was compared

| setting | `cleverly` | R `tmle3` |
| --- | --- | --- |
| estimator | ordinary TMLE, cross-fitting disabled | ordinary `tmle3` spec |
| datasets | generated in Python | the identical rows and all DGP covariates |
| nuisance regressions | GLM | corresponding GLM |
| treatment contrast | 1 versus 0 | 1 versus 0 |
| overlap | comfortable, no active bound | the same law |
| intervals | pointwise 95% Wald | pointwise 95% Wald |
| RR and OR scale | log, exponentiated | log, exponentiated |
| PAF scale | identity, from the PAF influence curve | negative-log-complement, transformed by `1 - exp(-x)` |
| ATT and ATC updater | default | the constrained one-dimensional updater the package's own tests exercise |

Both laws are point-treatment laws with complete outcomes. The bounded continuous-outcome law with
effect modification covers `ey1`, `ey0`, `ate`, `att`, `atc`, `ey_obs`, and `par`. The
binary-outcome law covers those plus `paf`, `rr`, and `or`.

### Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `atc` | average effect on the untreated | `cleverly` | -0.0026 to 0.0013 | 0.9506 | 0.9997 | pass |
| binary-outcome law | `atc` | average effect on the untreated | R `tmle3` | -0.0026 to 0.0013 | 0.9544 | 1.0049 | pass |
| binary-outcome law | `ate` | average treatment effect | `cleverly` | -0.0026 to 0.0012 | 0.9494 | 1.0067 | pass |
| binary-outcome law | `ate` | average treatment effect | R `tmle3` | -0.0026 to 0.0012 | 0.9487 | 1.0066 | pass |
| binary-outcome law | `att` | average effect on the treated | `cleverly` | -0.0028 to 0.0012 | 0.9506 | 1.0088 | pass |
| binary-outcome law | `att` | average effect on the treated | R `tmle3` | -0.0028 to 0.0012 | 0.9500 | 1.0141 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | `cleverly` | -0.000802 to 0.0020 | 0.9531 | 1.0102 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | R `tmle3` | -0.000802 to 0.0020 | 0.9531 | 1.0101 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | `cleverly` | -0.0016 to 0.0013 | 0.9425 | 0.9861 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | R `tmle3` | -0.0016 to 0.0013 | 0.9425 | 0.9861 | pass |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | `cleverly` | -0.000858 to 0.0012 | 0.9450 | 0.9919 | pass |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | R `tmle3` | -0.000858 to 0.0012 | 0.9450 | 0.9919 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | `cleverly` | -0.0095 to 0.0064 | 0.9500 | 1.0074 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | R `tmle3` | -0.0095 to 0.0064 | 0.9500 | 1.0074 | pass |
| binary-outcome law | `paf` | population attributable fraction | `cleverly` | -0.0030 to 0.0013 | 0.9494 | 1.0118 | pass |
| binary-outcome law | `paf` | population attributable fraction | R `tmle3` | -0.0029 to 0.0026 | 0.9487 | 1.0114 | pass |
| binary-outcome law | `par` | population attributable risk | `cleverly` | -0.0014 to 0.000586 | 0.9469 | 1.0082 | pass |
| binary-outcome law | `par` | population attributable risk | R `tmle3` | -0.0014 to 0.000587 | 0.9469 | 1.0081 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | `cleverly` | -0.0053 to 0.0033 | 0.9556 | 1.0117 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | R `tmle3` | -0.0053 to 0.0033 | 0.9556 | 1.0116 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | `cleverly` | -0.000400 to 0.000508 | 0.9463 | 0.9893 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | R `tmle3` | 0.000387 to 0.0013 | 0.9475 | 0.9957 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | `cleverly` | -0.000459 to 0.000392 | 0.9481 | 1.0015 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | R `tmle3` | -0.000442 to 0.000406 | 0.9500 | 1.0125 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | `cleverly` | -0.000602 to 0.000314 | 0.9394 | 0.9706 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | R `tmle3` | -0.0013 to -0.000431 | 0.9400 | 0.9726 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | `cleverly` | -0.000147 to 0.000743 | 0.9531 | 1.0146 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | R `tmle3` | -0.000147 to 0.000742 | 0.9537 | 1.0149 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | `cleverly` | -0.000260 to 0.000788 | 0.9519 | 1.0107 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | R `tmle3` | -0.000244 to 0.000802 | 0.9519 | 1.0122 | pass |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | `cleverly` | -0.000234 to 0.000779 | 0.9506 | 0.9908 | pass |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | R `tmle3` | -0.000234 to 0.000779 | 0.9506 | 0.9908 | pass |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | `cleverly` | -0.000327 to 0.000276 | 0.9344 | 0.9451 | pass |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | R `tmle3` | -0.000325 to 0.000277 | 0.9375 | 0.9526 | pass |
<!-- /generated -->

### Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `atc` | average effect on the untreated | 0.000023 | 0.0050 | 1.0066 | -0.0037 | pass |
| binary-outcome law | `ate` | average treatment effect | -6.992e-07 | 0.000157 | 1.0000 | 0.000625 | pass |
| binary-outcome law | `att` | average effect on the treated | -0.000022 | 0.0049 | 1.0067 | 0.000625 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | 9.290e-08 | 0.000029 | 1.0000 | 0 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | -4.085e-10 | 1.216e-07 | 1.0000 | 0 | pass |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | -8.722e-12 | 3.654e-09 | 1.0000 | 0 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | -5.723e-07 | 0.000014 | 1.0000 | 0 | pass |
| binary-outcome law | `paf` | population attributable fraction | -6.674e-07 | 0.000132 | 1.0002 | 0.000625 | pass |
| binary-outcome law | `par` | population attributable risk | -2.965e-07 | 0.000126 | 1.0002 | 0 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | -2.861e-07 | 0.000019 | 1.0000 | 0 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | -0.000783 | 0.7445 | 1.0088 | -0.0012 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | -0.000016 | 0.0161 | 1.0062 | -0.0019 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | 0.000744 | 0.6995 | 1.0019 | -0.000625 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | -6.584e-08 | 0.000064 | 1.0030 | -0.000625 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | -0.000015 | 0.0122 | 1.0032 | 0 | pass |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | -5.672e-13 | 4.814e-10 | 1.0000 | 0 | pass |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | -0.000001 | 0.0020 | 1.0045 | -0.0031 | pass |
<!-- /generated -->

### Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0104 to 0.0037, margin 0.0236 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.3420 to -0.3244, margin 0.0295 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0062 to 0.0069, margin 0.0220 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin | bias -0.0292 to -0.0072, margin 0.0369 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9420 to 0.9645, SE ratio 0.9679 to 1.0385 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000513, coverage 0.9194 to 0.9627, SE ratio 0.9777 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.000770, coverage 0.9223 to 0.9647, SE ratio 0.9913 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000285, coverage 0.9297 to 0.9698, SE ratio 1.0077 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5374 to -0.4725 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4998 to -0.4974 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0325, 0.0141 to 0.0627 | pass |
<!-- /generated -->

### Measured values

Every figure this section quotes is resolved by name and checked at the precision it is printed to.
Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results. Both resolve the same way, so moving a threshold in the code changes this
table rather than leaving it asserting a rule the study never applied.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | replications per law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 34 | implementation-estimand tests against truth |
| `independent_tests_passed` | 34 | of those, passing |
| `paired_tests_total` | 17 | scenario-estimand cells compared with `tmle3` |
| `paired_tests_passed` | 17 | of those, passing |
| `property_cells_total` | 12 | repeated-sampling property cells |
| `property_cells_passed` | 12 | of those, passing |
| `max_standardized_bias` | 0.1254 | largest bias, in empirical standard deviations |
| `min_coverage` | 0.9344 | lowest measured coverage of a nominal 95% interval |
| `min_coverage_ci_lower` | 0.9168 | lowest exact 99% coverage endpoint, against a floor of 0.90 |
| `min_se_ratio_ci_lower` | 0.9040 | lowest bootstrap SE-ratio endpoint, against a band of 0.80--1.20 |
| `max_se_ratio_ci_upper` | 1.0649 | highest bootstrap SE-ratio endpoint |
| `max_se_ratio_resolution` | 0.0960 | how far from 1.0 the widest SE-ratio interval still reaches |
| `max_margin_utilization` | 0.7445 | largest share of the paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0088 | largest one-sided RMSE-ratio bound, against a margin of 1.10 |
| `min_coverage_difference_lower` | -0.0081 | smallest one-sided coverage-difference bound, against -0.025 |
| `max_calibration_excess_upper` | 0.0129 | largest comparable SE-calibration-excess bound, against 0.05 |
| `summary_cells` | 34 | summary cells over both laws and both implementations |
| `cells_with_se_ratio_below_one` | 13 | of those, reporting a standard error below the empirical spread |
| `cells_with_coverage_below_nominal` | 17 | of those, covering below 95% |
| `summary[cleverly/continuous/par]:coverage` | 0.9344 | the study's lowest coverage, cleverly |
| `summary[tmle3/continuous/par]:coverage` | 0.9375 | the same cell in `tmle3` |
| `summary[cleverly/continuous/att]:bias` | -0.000144 | continuous ATT bias against known truth, cleverly |
| `summary[tmle3/continuous/att]:bias` | -0.000888 | the same, `tmle3` |
| `summary[cleverly/continuous/atc]:bias` | 0.000054 | continuous ATC bias against known truth, cleverly |
| `summary[tmle3/continuous/atc]:bias` | 0.000837 | the same, `tmle3` |
| `continuous_summary_cells` | 14 | continuous-law summary cells, both implementations |
| `continuous_cells_with_se_ratio_below_one` | 8 | of those, reporting a standard error below the empirical spread |
| `continuous_cells_with_coverage_below_nominal` | 7 | of those, covering below 95% |
| `properties[double_robustness/both_correct]:bias` | -0.0034 | bias with both nuisances correct |
| `properties[double_robustness/outcome_correct]:bias` | 0.000306 | bias with only the outcome regression correct |
| `properties[double_robustness/treatment_correct]:bias` | -0.0182 | bias with only the treatment mechanism correct |
| `properties[double_robustness/both_wrong]:bias` | -0.3332 | bias with both nuisances wrong; this is the negative control |
| `properties[double_robustness/both_wrong]:coverage` | 0.1967 | coverage of the same control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5045 | fitted log-log contraction rate of the sampling spread |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5374 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4725 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.4986 | the same rate for the mean reported standard error |
| `properties[root_n_and_efficiency/n_500]:se_ratio` | 0.9913 | SE calibration at n = 500 |
| `properties[root_n_and_efficiency/n_2000]:se_ratio` | 0.9777 | SE calibration at n = 2,000 |
| `properties[root_n_and_efficiency/n_8000]:se_ratio` | 1.0077 | SE calibration at n = 8,000 |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 1.0027 | SE calibration where both nuisances are correct |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9679 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0385 | its 99% upper endpoint |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9542 | coverage of the same cell |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0325 | rejection under a confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0627 | its 99% upper endpoint, against 0.05 + 0.05 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under a real effect; this is the positive control |
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

### Limitations

| limitation | what it means for use |
| --- | --- |
| The SE sanity band is a screen, not a calibration proof | The widest bootstrap SE-ratio interval still reaches `max_se_ratio_resolution` from 1.0. These 34 cells cannot rule out a systematic misstatement smaller than about a tenth. The `interval_calibration` cell carries that claim instead, and it excludes exactly that misstatement |
| PAF is compared on different native scales | Point performance and coverage are compared. Raw standard-error parity is not claimed. The two intervals share a first-order delta-method limit but need not share finite-sample endpoints |
| Continuous ATT and ATC use the most paired margin | `max_margin_utilization` of it at the widest, and both stay inside. Measured against known truth, `cleverly`'s absolute bias on those two estimands is an order of magnitude smaller than the constrained R path's, with essentially unchanged RMSE and coverage. The gate stays asymmetric so a future change reversing this fails the suite |
| The lowest coverage is continuous-law PAR | Both implementations sit low together and both clear the declared floor. What remains is a property of ordinary non-cross-fitted TMLE on this law, not of either implementation. The `root_n_and_efficiency` cells are the discriminator: a gap that shrinks with `n` is finite-sample noise, and one that persisted would be a wrong variance formula |
| Interval calibration shows no systematic direction | The reported standard error falls below the empirical spread in `cells_with_se_ratio_below_one` of `summary_cells` cells and above it in the rest. An earlier run at a quarter of the replication count had every continuous-law cell below one and was read as systematic understatement. Raising the replication count dissolved it |
| The row is bounded to ordinary TMLE on two laws | It does not establish cross-fitting, repeated cross-fitting, CV-TMLE evaluation, simultaneous intervals, bootstrap inference, missing outcomes, weights, clustering, strata, multi-valued treatment, or flexible learners. Each is a separate method or composition and needs its own study |

### Reproduction

The container pins R 4.5.2 by digest, `tmle3` at `ed72f8a`, and `sl3` at
[`0e8f236`](https://github.com/tlverse/sl3/tree/0e8f2365bcbe54010b8120c04a7a2dcfc8119227). The
[fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/README.md)
gives the regeneration commands.
[`manifest.json`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/manifest.json)
records the configuration, every declared margin, the `cleverly` version and commit, the interpreter
and library versions, and a hash of every published result and study module. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/replicates.csv.gz),
[performance tests](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/performance-tests.csv),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/equivalence.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/properties.csv)
carry every row the tables above publish.

R uses its public specifications except for ATT and ATC. The public ATT convenience path fails to
converge on a small fraction of bounded-continuous samples, so both use the constrained path the
pinned package's own tests exercise. That path is fixed for all replications. The R side aborts on
any failed replication rather than dropping it, and the property study refuses to summarize a run
that lost one.

## Stacked point-treatment CV-TMLE

This study validates `cleverly`'s default cross-validated point-treatment construction, described
by Levy (2018). Nuisance predictions are out of fold, one targeting regression is fitted over the
stacked validation rows, and the updated regression is evaluated over the whole sample. Zheng and
van der Laan (2011) supply the wider CV-TMLE framework. The source boundary is mapped in
[CV-TMLE and cross-fitting](cv-tmle.md#the-algorithm-as-implemented).

### What was compared

| setting | `cleverly` | R `tmle3` CV-TMLE |
| --- | --- | --- |
| construction | stacked update, whole-sample plug-in evaluation | `tmle3_Update(cvtmle = TRUE)` |
| folds | treatment-stratified ten-fold, generated in Python | the identical validation indices, rebuilt with `origami` and asserted on the task |
| nuisance learners | GLM | corresponding GLM wrapped in `Lrnr_cv` |
| propensity bounds | 0.025 to 0.975 | 0.025 to 0.975 |
| intervals | pointwise 95% Wald | pointwise 95% Wald |
| PAF scale | identity, from the PAF influence curve | negative-log-complement, transformed |

Supplying the exact row-to-fold assignment is what makes this a fold-matched comparison. A common
seed or fold count is not enough when splitters or dependency versions differ. The R runner aborts
the whole study on any failed fit, changed fold, missing estimand, or dropped replication.

### Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `atc` | average effect on the untreated | `cleverly` stacked CV-TMLE | -0.0043 to -0.000378 | 0.9456 | 1.0139 | pass |
| binary-outcome law | `atc` | average effect on the untreated | R `tmle3` CV-TMLE | -0.0042 to -0.000352 | 0.9469 | 1.0195 | pass |
| binary-outcome law | `ate` | average treatment effect | `cleverly` stacked CV-TMLE | -0.0040 to -0.000199 | 0.9494 | 1.0163 | pass |
| binary-outcome law | `ate` | average treatment effect | R `tmle3` CV-TMLE | -0.0040 to -0.000201 | 0.9506 | 1.0164 | pass |
| binary-outcome law | `att` | average effect on the treated | `cleverly` stacked CV-TMLE | -0.0039 to 0.000098 | 0.9519 | 1.0216 | pass |
| binary-outcome law | `att` | average effect on the treated | R `tmle3` CV-TMLE | -0.0039 to 0.000081 | 0.9525 | 1.0257 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | `cleverly` stacked CV-TMLE | -0.000696 to 0.0021 | 0.9556 | 1.0076 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | R `tmle3` CV-TMLE | -0.000696 to 0.0021 | 0.9556 | 1.0076 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | `cleverly` stacked CV-TMLE | -0.0028 to 0.000051 | 0.9375 | 0.9906 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | R `tmle3` CV-TMLE | -0.0028 to 0.000051 | 0.9375 | 0.9906 | pass |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | `cleverly` stacked CV-TMLE | -0.0013 to 0.000802 | 0.9456 | 0.9815 | pass |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | R `tmle3` CV-TMLE | -0.0013 to 0.000802 | 0.9456 | 0.9815 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | `cleverly` stacked CV-TMLE | -0.0151 to 0.000844 | 0.9494 | 1.0167 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | R `tmle3` CV-TMLE | -0.0151 to 0.000844 | 0.9494 | 1.0166 | pass |
| binary-outcome law | `paf` | population attributable fraction | `cleverly` stacked CV-TMLE | -0.0039 to 0.000426 | 0.9500 | 1.0234 | pass |
| binary-outcome law | `paf` | population attributable fraction | R `tmle3` CV-TMLE | -0.0040 to 0.0014 | 0.9525 | 1.0233 | pass |
| binary-outcome law | `par` | population attributable risk | `cleverly` stacked CV-TMLE | -0.0020 to 0.000061 | 0.9500 | 1.0209 | pass |
| binary-outcome law | `par` | population attributable risk | R `tmle3` CV-TMLE | -0.0020 to 0.000061 | 0.9506 | 1.0212 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | `cleverly` stacked CV-TMLE | -0.0078 to 0.000826 | 0.9475 | 1.0206 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | R `tmle3` CV-TMLE | -0.0078 to 0.000826 | 0.9475 | 1.0205 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | `cleverly` stacked CV-TMLE | -0.000556 to 0.000353 | 0.9494 | 0.9995 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | R `tmle3` CV-TMLE | 0.000163 to 0.0011 | 0.9500 | 1.0061 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | `cleverly` stacked CV-TMLE | -0.000502 to 0.000337 | 0.9494 | 1.0260 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | R `tmle3` CV-TMLE | -0.000500 to 0.000335 | 0.9506 | 1.0391 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | `cleverly` stacked CV-TMLE | -0.000483 to 0.000426 | 0.9444 | 0.9887 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | R `tmle3` CV-TMLE | -0.0012 to -0.000308 | 0.9406 | 0.9904 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | `cleverly` stacked CV-TMLE | -0.000446 to 0.000473 | 0.9469 | 0.9851 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | R `tmle3` CV-TMLE | -0.000438 to 0.000480 | 0.9463 | 0.9853 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | `cleverly` stacked CV-TMLE | -0.000593 to 0.000456 | 0.9544 | 1.0142 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | R `tmle3` CV-TMLE | -0.000592 to 0.000456 | 0.9556 | 1.0159 | pass |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | `cleverly` stacked CV-TMLE | -0.000462 to 0.000556 | 0.9431 | 0.9862 | pass |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | R `tmle3` CV-TMLE | -0.000462 to 0.000556 | 0.9431 | 0.9862 | pass |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | `cleverly` stacked CV-TMLE | -0.000261 to 0.000327 | 0.9387 | 0.9762 | pass |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | R `tmle3` CV-TMLE | -0.000289 to 0.000297 | 0.9394 | 0.9852 | pass |
<!-- /generated -->

### Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `atc` | average effect on the untreated | -0.000037 | 0.0081 | 1.0070 | -0.0012 | pass |
| binary-outcome law | `ate` | average treatment effect | 0.000002 | 0.000412 | 1.0001 | -0.0013 | pass |
| binary-outcome law | `att` | average effect on the treated | 0.000010 | 0.0021 | 1.0054 | -0.000625 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | 7.188e-08 | 0.000022 | 1.0000 | 0 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | 8.047e-08 | 0.000024 | 1.0000 | 0 | pass |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | -7.174e-12 | 2.974e-09 | 1.0000 | 0 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | 6.970e-09 | 1.750e-07 | 1.0000 | 0 | pass |
| binary-outcome law | `paf` | population attributable fraction | -3.430e-07 | 0.000068 | 1.0007 | -0.0025 | pass |
| binary-outcome law | `par` | population attributable risk | -1.467e-07 | 0.000062 | 1.0007 | -0.000625 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | -2.158e-08 | 0.000001 | 1.0000 | 0 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | -0.000715 | 0.6790 | 1.0128 | -0.000625 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | 2.801e-07 | 0.000288 | 1.0082 | -0.0013 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | 0.000733 | 0.6941 | 1.0041 | 0.0037 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | -0.000007 | 0.0069 | 1.0032 | 0.000625 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | -6.210e-07 | 0.000509 | 1.0039 | -0.0012 | pass |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | -1.684e-13 | 1.421e-10 | 1.0000 | 0 | pass |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | 0.000029 | 0.0422 | 1.0072 | -0.000625 | pass |
<!-- /generated -->

### Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.5298 to 0.6400 | pass |
| `crossfit_overfitting` | `stacked_cvtmle` | positive | stacked CV-TMLE with a flexible learner | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9094 to 1.0843 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0100 to 0.0038, margin 0.0231 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.3437 to -0.3259, margin 0.0299 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0064 to 0.0067, margin 0.0219 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin | bias -0.0305 to -0.0086, margin 0.0366 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9430 to 0.9652, SE ratio 0.9738 to 1.0450 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000506, coverage 0.9238 to 0.9657, SE ratio 0.9823 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.000654, coverage 0.9252 to 0.9667, SE ratio 1.0128 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000293, coverage 0.9326 to 0.9717, SE ratio 1.0087 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5384 to -0.4737 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5080 to -0.5055 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0325, 0.0141 to 0.0627 | pass |
<!-- /generated -->

### Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | replications per law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 34 | implementation-estimand tests against truth |
| `independent_tests_passed` | 34 | of those, passing |
| `paired_tests_total` | 17 | scenario-estimand paired tests |
| `paired_tests_passed` | 17 | of those, passing |
| `property_cells_total` | 14 | repeated-sampling property cells |
| `property_cells_passed` | 14 | of those, passing |
| `max_standardized_bias` | 0.1082 | largest absolute bias in empirical standard deviations |
| `min_coverage` | 0.9375 | lowest measured coverage of a nominal 95% interval |
| `min_coverage_ci_lower` | 0.9203 | lowest exact 99% coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9329 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0884 | highest bootstrap SE-ratio endpoint |
| `max_se_ratio_resolution` | 0.0884 | farthest the widest SE-ratio interval reaches from 1.0 |
| `max_margin_utilization` | 0.6941 | largest share of a paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0128 | largest one-sided RMSE-ratio bound, against 1.10 |
| `min_coverage_difference_lower` | -0.0081 | smallest one-sided coverage-difference bound, against -0.025 |
| `max_calibration_excess_upper` | 0.0116 | largest SE-calibration-excess bound, against 0.05 |
| `properties[double_robustness/outcome_correct]:bias` | 0.000149 | bias with only the outcome nuisance correct |
| `properties[double_robustness/treatment_correct]:bias` | -0.0196 | bias with only the treatment nuisance correct |
| `properties[double_robustness/both_wrong]:bias` | -0.3348 | both-wrong negative control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5053 | fitted log-log sampling-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5384 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4737 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.5067 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 1.0077 | SE calibration where both nuisances are correct |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9738 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0450 | its 99% upper endpoint |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9550 | coverage of the same cell |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0325 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0627 | its 99% upper endpoint, against 0.10 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the positive control |
| `properties[crossfit_overfitting/stacked_cvtmle]:coverage` | 0.8950 | coverage with cross-fitted tree predictions |
| `properties[crossfit_overfitting/stacked_cvtmle]:se_ratio` | 0.9880 | SE calibration with cross-fitting |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.6500 | coverage with the deliberately in-sample tree |
| `properties[crossfit_overfitting/in_sample_control]:se_ratio` | 0.5792 | SE calibration of that control |
| `properties[crossfit_overfitting/stacked_cvtmle]:coverage_gain_ci_lower` | 0.1875 | paired 99% lower bound for coverage gained over the control |
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
| `margin:overfit_se_floor` | 0.8500 | SE ratio the cross-fit arm must restore |
| `margin:overfit_control_ceiling` | 0.7500 | ceiling the in-sample control's upper bound must stay below |
| `margin:overfit_coverage_gain` | 0.1500 | coverage cross-fitting must buy over the in-sample control |

### Limitations

| limitation | what it means for use |
| --- | --- |
| PAF is compared on different native scales | `cleverly` reports its fraction-scale influence curve and `tmle3` transforms a negative-log-complement interval. Point performance and coverage are compared. Raw standard errors and finite-sample endpoints on those scales are not declared equivalent |
| The cross-fit overfitting cells are relative evidence | Held-out predictions must restore the SE ratio to its band, the in-sample control's upper bound must stay below 0.75, and the paired coverage gain must clear its floor. The measured cross-fitted coverage is evidence of relative recovery and calibrated influence-curve scale. It is not a separate absolute coverage claim; the primary GLM study carries that gate |
| Continuous-law ATT and ATC use the most similarity margin | Both stay inside the predeclared bound, and the RMSE, coverage, and calibration bounds stay well inside their non-inferiority margins. These are the finite-sample cells that use the most evidence budget, not tuned exceptions |
| The row is bounded to one ten-fold split | It does not establish repeated or nested cross-fitting, fold-evaluated or fold-specific-epsilon CV-TMLE, simultaneous or bootstrap intervals, missing outcomes, weights, clusters, strata, multi-valued treatment, broad learner-library selection, or severe practical-positivity behaviour |

### Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/README.md)
gives the full and smoke regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/manifest.json)
records every margin, seed, package pin, source hash, and result hash. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/equivalence.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/properties.csv)
carry every published row.

## Fold-evaluated point-treatment CV-TMLE

This study validates `cleverly`'s fold-evaluated CV-TMLE report: treatment-stratified ten-fold
nuisance fitting, one pooled targeting update, equal-fold plug-in evaluation, and the
cross-validated influence-curve variance. It is separate from the stacked row above because
averaging fold reports, rather than evaluating the updated regression over the whole sample, is a
genuine finite-sample method choice. The construction and its source boundary are mapped in
[CV-TMLE and cross-fitting](cv-tmle.md#the-algorithm-as-implemented).

**No canonical implementation is compared.** No maintained package ships this construction, so the
study rests on the accuracy and theory-property questions alone. A zero-row equivalence artifact
records that absence rather than borrowing the stacked R comparison.

### What was compared

| setting | `cleverly` |
| --- | --- |
| construction | ten-fold nuisance fitting, pooled update, equal-fold plug-in evaluation |
| variance | cross-validated influence curve |
| estimands | `ey1`, `ey0`, `ate`, `att`, `atc` |
| laws | the same binary and bounded-continuous laws the ordinary TMLE study uses |
| nuisance learners | corresponding logistic and linear GLM |
| propensity bounds | 0.025 to 0.975 |
| intervals | pointwise 95% identity-scale Wald |

### Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `atc` | average effect on the untreated | `cleverly` fold-evaluated CV-TMLE | -0.000879 to 0.0031 | 0.9456 | 0.9987 | pass |
| binary-outcome law | `ate` | average treatment effect | `cleverly` fold-evaluated CV-TMLE | -0.0011 to 0.0028 | 0.9506 | 0.9997 | pass |
| binary-outcome law | `att` | average effect on the treated | `cleverly` fold-evaluated CV-TMLE | -0.0015 to 0.0026 | 0.9550 | 1.0048 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | `cleverly` fold-evaluated CV-TMLE | -0.0019 to 0.000936 | 0.9525 | 1.0093 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | `cleverly` fold-evaluated CV-TMLE | -0.0011 to 0.0018 | 0.9575 | 1.0155 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | `cleverly` fold-evaluated CV-TMLE | -0.000360 to 0.000559 | 0.9513 | 0.9866 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | `cleverly` fold-evaluated CV-TMLE | -0.000264 to 0.000593 | 0.9500 | 1.0034 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | `cleverly` fold-evaluated CV-TMLE | -0.000186 to 0.000735 | 0.9375 | 0.9756 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | `cleverly` fold-evaluated CV-TMLE | -0.000702 to 0.000203 | 0.9519 | 0.9967 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | `cleverly` fold-evaluated CV-TMLE | -0.000603 to 0.000433 | 0.9537 | 1.0243 | pass |
<!-- /generated -->

### Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `fold_evaluated_cvtmle` | positive | fold-evaluated CV-TMLE with a flexible learner | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9091 to 1.0908 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.5308 to 0.6408 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0100 to 0.0038, margin 0.0231 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.3437 to -0.3259, margin 0.0299 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0064 to 0.0067, margin 0.0219 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin | bias -0.0305 to -0.0086, margin 0.0366 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9430 to 0.9652, SE ratio 0.9730 to 1.0456 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000506, coverage 0.9238 to 0.9657, SE ratio 0.9825 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.000654, coverage 0.9252 to 0.9667, SE ratio 1.0139 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000293, coverage 0.9326 to 0.9717, SE ratio 1.0088 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5374 to -0.4732 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5084 to -0.5058 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0325, 0.0141 to 0.0627 | pass |
<!-- /generated -->

### Measured values

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
| `max_standardized_bias` | 0.0384 | largest absolute bias in empirical standard deviations |
| `min_coverage` | 0.9375 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9203 | lowest exact 99% coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9303 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0737 | highest bootstrap SE-ratio endpoint |
| `properties[double_robustness/outcome_correct]:bias` | 0.000149 | bias with only the outcome nuisance correct |
| `properties[double_robustness/treatment_correct]:bias` | -0.0196 | bias with only the treatment nuisance correct |
| `properties[double_robustness/both_wrong]:bias` | -0.3348 | both-wrong negative control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5053 | fitted log-log sampling-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5374 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4732 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.5071 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 1.0080 | SE calibration where both nuisances are correct |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9730 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0456 | its 99% upper endpoint |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9550 | coverage of the same cell |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0325 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0627 | its 99% upper endpoint, against 0.10 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the positive control |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:coverage` | 0.8950 | coverage with cross-fitted tree predictions |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:se_ratio` | 0.9910 | SE calibration with cross-fitting |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.6500 | coverage with the deliberately in-sample tree |
| `properties[crossfit_overfitting/in_sample_control]:se_ratio` | 0.5792 | SE calibration of that control |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:coverage_gain_ci_lower` | 0.1875 | paired 99% lower bound for coverage gained over the control |
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
| `margin:overfit_se_floor` | 0.8500 | SE ratio the cross-fit arm must restore |
| `margin:overfit_control_ceiling` | 0.7500 | ceiling the in-sample control's upper bound must stay below |
| `margin:overfit_coverage_gain` | 0.1500 | coverage cross-fitting must buy over the in-sample control |

### Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | The row rests on accuracy against known truth and on the theory properties. It is not parity evidence for stacked R CV-TMLE, and it does not inherit the stacked row's comparison |
| The cross-fit overfitting cells are relative evidence | A fully grown regression tree is fitted on the nonlinear law twice, once with held-out predictions and once in sample, on the identical 400 samples of size 500. The cross-fitted cell's evidence is restored SE calibration and a load-bearing improvement over the control. The primary GLM study carries the absolute coverage gate |
| The row is bounded to one fixed ten-fold assignment per sample | It does not establish repeated or nested cross-fitting, a fold-specific targeting epsilon, simultaneous or bootstrap intervals, missing outcomes, weights, clusters, strata, multi-valued treatment, ratio estimands, observed-risk functionals, or behaviour under severe practical-positivity violations |

### Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/manifest.json)
records the primary and control samples, every margin and seed, the exact estimator configuration,
and the source and result hashes. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/replicates.csv.gz)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/properties.csv)
carry every published row.

## Selector-based point-treatment C-TMLE

This study validates `cleverly`'s greedy, ordered, and discrete C-TMLE selectors. The canonical
comparison uses R [`ctmle`](https://github.com/jucheng1992/ctmle) 0.1.2 at pinned commit
[`18de559`](https://github.com/jucheng1992/ctmle/tree/18de559f47dc1286617350a0668391e80e1dbf7c).
That package is the maintained comparator for these selector entry points. It is not a tlverse
package, and the tlverse comparison applies only to the outcome-adaptive study below. The theory
is van der Laan and Gruber (2010).

### What was compared

| setting | `cleverly` | R `ctmle` |
| --- | --- | --- |
| strategies | `greedy`, `ordered`, `discrete` | `ctmleGeneral`, and `ctmleDiscrete(preOrder = TRUE)` for both ordered and discrete |
| datasets | binary-outcome samples generated in Python with their exact ATE | the identical rows and all three DGP covariates |
| selector folds | treatment-stratified five-fold, taken off the `cleverly` fit that produced the subject row | the same assignment, asserted to be a partition before selecting |
| nuisance regressions | corresponding logistic GLM | corresponding logistic GLM |
| propensity bounds | 0.025 to 0.975 | 0.025 to 0.975 |
| selector loss | unpenalized | unpenalized |
| cross-fitting | disabled for the comparison | not applicable |

`cleverly`'s default penalty follows the published trace-plus-bias criterion. It is not presented
as numerical parity with R's implementation-specific adjustment. The property study exercises a
public nested cross-fit configuration instead: five outer folds, three selection folds, and two
inner folds, with the penalty on.

### Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law, discrete selector | `ate` | average treatment effect | `cleverly` selector-based C-TMLE | -0.0022 to 0.0018 | 0.9413 | 0.9448 | pass |
| binary-outcome law, discrete selector | `ate` | average treatment effect | R `ctmle` | -0.0021 to 0.0018 | 0.9400 | 0.9383 | pass |
| binary-outcome law, greedy selector | `ate` | average treatment effect | `cleverly` selector-based C-TMLE | -0.0018 to 0.0021 | 0.9500 | 0.9591 | pass |
| binary-outcome law, greedy selector | `ate` | average treatment effect | R `ctmle` | -0.0018 to 0.0021 | 0.9475 | 0.9452 | pass |
| binary-outcome law, ordered selector | `ate` | average treatment effect | `cleverly` selector-based C-TMLE | -0.0010 to 0.0028 | 0.9413 | 0.9636 | pass |
| binary-outcome law, ordered selector | `ate` | average treatment effect | R `ctmle` | -0.0010 to 0.0028 | 0.9387 | 0.9616 | pass |
<!-- /generated -->

### Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law, discrete selector | `ate` | average treatment effect | -0.000062 | 0.0189 | 1.0030 | 0.0013 | pass |
| binary-outcome law, greedy selector | `ate` | average treatment effect | 0.000053 | 0.0168 | 1.0051 | 0.0025 | pass |
| binary-outcome law, ordered selector | `ate` | average treatment effect | 0.000004 | 0.0011 | 1.0048 | 0.0025 | pass |
<!-- /generated -->

### Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0047 to 0.0071, margin 0.0198 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin | bias 0.1028 to 0.1226, margin 0.0333 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0075 to 0.0038, margin 0.0191 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin | bias 0.0047 to 0.0154, margin 0.0179 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9305 to 0.9552, SE ratio 0.9579 to 1.0340 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000661, coverage 0.9107 to 0.9565, SE ratio 0.9623 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.0019, coverage 0.9311 to 0.9708, SE ratio 0.9828 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000211, coverage 0.9252 to 0.9667, SE ratio 0.9622 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5270 to -0.4640 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5043 to -0.5019 | pass |
| `selector_necessity` | `collaborative` | positive | the selector chooses its own mechanism path | bias interval inside the equivalence margin | bias 0.0043 to 0.0199, margin 0.0214 | pass |
| `selector_necessity` | `empty_control` | control | the selector is forced to stop at an empty path | bias interval must fall entirely outside the margin | bias 0.7865 to 0.8046, margin 0.0247 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0275, 0.0109 to 0.0561 | pass |
<!-- /generated -->

### Measured values

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
| `property_cells_passed` | 14 | of those, passing |
| `max_standardized_bias` | 0.0426 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9387 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9136 | lowest exact 99% coverage endpoint |
| `min_se_ratio_ci_lower` | 0.8850 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0342 | highest bootstrap SE-ratio endpoint |
| `max_margin_utilization` | 0.0189 | largest share of the paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0051 | largest paired RMSE-ratio upper bound |
| `min_coverage_difference_lower` | -0.0063 | smallest paired coverage-difference lower bound |
| `max_calibration_excess_upper` | 0.0122 | largest paired SE-calibration-excess upper bound |
| `properties[double_robustness/both_correct]:standardized_bias` | 0.0155 | both nuisances correct |
| `properties[double_robustness/outcome_correct]:standardized_bias` | -0.0242 | only the outcome nuisance correct |
| `properties[double_robustness/treatment_correct]:standardized_bias` | 0.1408 | only the treatment nuisance correct |
| `properties[double_robustness/treatment_correct]:n` | 2000 | observations that leg needs to resolve its remainder |
| `properties[double_robustness/both_wrong]:standardized_bias` | 0.8464 | both-wrong negative control |
| `properties[selector_necessity/collaborative]:rmse_ratio` | 0.1077 | collaborative RMSE divided by the empty-path control RMSE |
| `properties[selector_necessity/collaborative]:se_ratio` | 1.2539 | reported SE over empirical spread in that cell; see below |
| `properties[root_n_rate/empirical_sd]:slope` | -0.4954 | fitted empirical-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5270 | its 99% lower endpoint, against a band of -0.625 to -0.375 |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4640 | its 99% upper endpoint, which must also exclude -0.25 |
| `properties[root_n_rate/reported_se]:slope` | -0.5030 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9437 | calibration-cell coverage |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 0.9942 | calibration-cell SE ratio |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9579 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0340 | its 99% upper endpoint |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0275 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0561 | its 99% upper endpoint, against 0.05 + 0.05 |
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
| `margin:selector_rmse_ratio` | 0.5000 | maximum collaborative-to-empty RMSE ratio |

### Limitations

| limitation | what it means for use |
| --- | --- |
| Two of the three strategies reach the same R entry point | R `ctmle` has one pre-ordered selector, so `ordered` and `discrete` are both compared against `ctmleDiscrete(preOrder = TRUE)`. The correspondence is earned: the `discrete` candidate list is exactly the nested prefix ladder that mode enumerates. An arbitrary candidate list therefore has no reference here, and the row carries two reference constructions on three separate draws |
| The two sides select differently even where they agree | `cleverly` refits the outcome regression inside two nested folds within each selection fold. R scores every fold against one full-sample `Q`. Agreement here is evidence about the C-TMLE machinery on a law where the selector's choice is stable, not about the selection rule. That the search is load-bearing is established by `selector_necessity` and by the unit tests |
| No cell asks for calibrated inference while selection is load-bearing | The forced-selection cell claims the RMSE ratio and only that. Its reported standard error is `properties[selector_necessity/collaborative]:se_ratio` of the empirical spread, which is conservative and outside the sanity screen. That is deliberate: the instrument law is built to make selection necessary, not to be a setting where the influence-curve variance is efficient. But `interval_calibration` is measured where both nuisances are correct and the search has nothing to do, so the gap is real |
| One robustness cell is sized differently from its siblings | `treatment_correct` runs at `properties[double_robustness/treatment_correct]:n` observations where the other three run at 700. It is the leg that leans on inverse weighting, and at 700 its `O(n^-1)` remainder is about 0.28 empirical standard deviations. Raising `n` resolves the remainder against an unchanged margin. The margin was not moved after seeing it |
| The default ordering and two strategies lack their own property evidence | The `ordered` cell pins an explicit covariate order, so the default `preorder="logistic"` ordering is exercised by neither half of this row. The property cells all run the default `greedy` search, so `ordered` and `discrete` have parity evidence without repeated-sampling evidence |
| The parity claim is narrow | It is binary, two-arm, complete-outcome, GLM, non-cross-fitted, unpenalized ATE only. The row does not establish parity for the default penalty or nested cross-fitting, ratios or arm means, missing outcomes, weights, clusters, strata, multi-valued treatment, flexible learner libraries, simultaneous or bootstrap intervals, or severe practical-positivity behaviour |

### Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/README.md)
and [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/manifest.json)
record the provenance and the regeneration commands. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/equivalence.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/properties.csv)
carry every published row.

## Outcome-adaptive point-treatment C-TMLE

This study validates `CTMLE(strategy="oat")`, whose treatment mechanism is fitted on the vector of
arm-specific outcome-regression predictions rather than on a selected subset of the original
covariates. The canonical comparison uses archived tlverse
[`ctmle3`](https://github.com/tlverse/ctmle3) 0.1.0 at commit
[`a4ea77b`](https://github.com/tlverse/ctmle3/tree/a4ea77b07747dfee9b2eecb9cbca88262e0559ea), with
contemporaneous `tmle3` at
[`3a61005`](https://github.com/tlverse/tmle3/tree/3a610058cd89c17bb417c15fc891254388787f33) and
`sl3` at
[`821ca89`](https://github.com/tlverse/sl3/tree/821ca890cb8701fdb59f823e28c6356e50d092bc). The
theory is Ju et al. (2019).

### What was compared

| setting | `cleverly` | R `ctmle3` |
| --- | --- | --- |
| construction | outcome-adaptive, non-cross-fitted | the archived non-cross-fitted OAT construction |
| datasets | binary samples generated in Python with exact truths | the identical rows |
| outcome regression | three-covariate logistic GLM | the same three-covariate logistic GLM |
| estimands | both treatment-specific means, ATE, marginal RR, marginal OR | the same |
| ratio scale | log, delta-method curves | log, delta-method curves |
| propensity bounds | 0.025 to 0.975 | no truncation of its own |

The bound difference is declared rather than hidden. On this law the true propensity spans 0.085 to
0.904, so the bound is never active and no paired comparison is affected. The manifest's `g_bounds`
entry therefore describes the subject's setting and not a shared one.

### Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `ate` | average treatment effect | `cleverly` outcome-adaptive C-TMLE | -0.0031 to 0.0024 | 0.9425 | 0.9787 | pass |
| binary-outcome law | `ate` | average treatment effect | R `ctmle3` | -0.0032 to 0.0024 | 0.9425 | 0.9785 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | `cleverly` outcome-adaptive C-TMLE | -0.000870 to 0.0031 | 0.9625 | 1.0049 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | R `ctmle3` | -0.000869 to 0.0031 | 0.9625 | 1.0049 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | `cleverly` outcome-adaptive C-TMLE | -0.0013 to 0.0028 | 0.9475 | 0.9814 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | R `ctmle3` | -0.0013 to 0.0028 | 0.9475 | 0.9813 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | `cleverly` outcome-adaptive C-TMLE | -0.0117 to 0.0115 | 0.9437 | 0.9808 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | R `ctmle3` | -0.0117 to 0.0114 | 0.9437 | 0.9806 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | `cleverly` outcome-adaptive C-TMLE | -0.0070 to 0.0055 | 0.9475 | 0.9865 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | R `ctmle3` | -0.0070 to 0.0055 | 0.9463 | 0.9864 | pass |
<!-- /generated -->

### Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `ate` | average treatment effect | 0.000005 | 0.0011 | 1.0003 | 0 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | 3.839e-08 | 0.000012 | 1.0005 | 0 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | 0.000005 | 0.0016 | 1.0002 | 0 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | 0.000043 | 0.0011 | 1.0004 | 0 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | 0.000014 | 0.000940 | 1.0004 | 0.0012 | pass |
<!-- /generated -->

### Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `cross_fitted_oat` | positive | outcome-adaptive C-TMLE with cross-fitted nuisances and a flexible learner | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9826 to 1.1711 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.5116 to 0.6100 | pass |
| `generated_design` | `estimated` | control | the same design is estimated from the data, as a real fit does | the SE-ratio deficit must reach the declared shortfall | SE ratio 0.9042 to 1.0047 | pass |
| `generated_design` | `oracle_design` | positive | the outcome-adaptive design is supplied rather than estimated | SE ratio interval inside the calibration band | SE ratio 0.9396 to 1.0430 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9337 to 0.9578, SE ratio 0.9506 to 1.0250 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `robustness_contract` | `outcome_correct` | positive | the outcome regression is correct and the mechanism is not | bias interval inside the equivalence margin | bias -0.0048 to 0.0075, margin 0.0206 | pass |
| `robustness_contract` | `outcome_wrong` | control | the outcome regression is misspecified | bias interval must fall entirely outside the margin | bias -0.3084 to -0.2904, margin 0.0302 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0027, coverage 0.9136 to 0.9585, SE ratio 0.9667 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.0019, coverage 0.9194 to 0.9627, SE ratio 0.9687 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0011, coverage 0.9238 to 0.9657, SE ratio 0.9542 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5287 to -0.4651 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5033 to -0.5011 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0725, 0.0509 to 0.0994 | pass |
<!-- /generated -->

### Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | replications of the binary law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 10 | implementation-estimand tests against truth |
| `independent_tests_passed` | 10 | of those, passing |
| `paired_tests_total` | 5 | paired estimand tests |
| `paired_tests_passed` | 5 | of those, passing |
| `property_cells_total` | 14 | independent property cells |
| `property_cells_passed` | 14 | of those, passing |
| `max_standardized_bias` | 0.0513 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9425 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9179 | lowest exact 99% coverage endpoint |
| `min_se_ratio_ci_lower` | 0.9211 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0769 | highest bootstrap SE-ratio endpoint |
| `max_margin_utilization` | 0.0016 | largest share of the paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0005 | largest paired RMSE-ratio upper bound |
| `min_coverage_difference_lower` | 0 | smallest paired coverage-difference lower bound |
| `max_calibration_excess_upper` | 0.000592 | largest paired SE-calibration-excess upper bound |
| `properties[robustness_contract/outcome_correct]:standardized_bias` | 0.0166 | outcome-correct OAT bias |
| `properties[robustness_contract/outcome_wrong]:standardized_bias` | -2.4768 | outcome-wrong negative control |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5287 | 99% lower endpoint of the empirical-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4651 | its upper endpoint, which must also exclude -0.25 |
| `properties[root_n_rate/empirical_sd]:slope` | -0.4967 | fitted empirical-spread rate |
| `properties[root_n_rate/reported_se]:slope` | -0.5022 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9467 | calibration-cell coverage |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 0.9870 | calibration-cell SE ratio |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9506 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0250 | its 99% upper endpoint |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0725 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0994 | its 99% upper endpoint, against 0.05 + 0.05 |
| `properties[type_i_error/sharp_null]:coverage_ci_lower` | 0.9006 | its exact 99% coverage endpoint, against a floor of 0.90 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the power control |
| `properties[crossfit_overfitting/cross_fitted_oat]:coverage` | 0.9325 | coverage with cross-fitted tree predictions |
| `properties[crossfit_overfitting/cross_fitted_oat]:standardized_bias` | -0.4361 | that arm's bias, in empirical standard deviations |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.6225 | coverage with the in-sample tree control |
| `properties[crossfit_overfitting/cross_fitted_oat]:se_ratio` | 1.0681 | SE ratio with cross-fitting |
| `properties[crossfit_overfitting/in_sample_control]:se_ratio` | 0.5565 | SE ratio without cross-fitting |
| `properties[crossfit_overfitting/cross_fitted_oat]:coverage_gain_ci_lower` | 0.2525 | paired 99% lower bound for coverage gained by cross-fitting |
| `properties[generated_design/oracle_design]:se_ratio` | 0.9892 | SE ratio with the design pinned at the truth |
| `properties[generated_design/oracle_design]:se_ratio_ci_lower` | 0.9396 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[generated_design/oracle_design]:se_ratio_ci_upper` | 1.0430 | its 99% upper endpoint |
| `properties[generated_design/estimated]:se_ratio` | 0.9525 | the same ratio with the design estimated |
| `properties[generated_design/estimated]:se_ratio_ci_lower` | 0.9042 | its 99% lower endpoint |
| `properties[generated_design/estimated]:se_ratio_ci_upper` | 1.0047 | its 99% upper endpoint |
| `properties[generated_design/estimated]:se_ratio_deficit_lower` | -0.0546 | paired 99% lower endpoint for estimated minus pinned |
| `properties[generated_design/estimated]:se_ratio_deficit_upper` | -0.0191 | its upper endpoint, which must clear the floor below |
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
| `margin:overfit_se_floor` | 0.8500 | cross-fit SE-ratio lower limit |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample SE-ratio upper limit |
| `margin:overfit_coverage_gain` | 0.1500 | cross-fit coverage-gain lower limit |
| `margin:generated_design_deficit` | 0.0100 | smallest paired SE-ratio deficit the control must establish |

### Limitations

| limitation | what it means for use |
| --- | --- |
| OAT has a narrower robustness contract than selector C-TMLE | With the outcome regression correct, the bias interval must fit inside the equivalence margin. With it wrong, the control must be discriminated outside it. No treatment-correct-only claim is made, because OAT's mechanism is a projection on the generated outcome-regression design rather than a fit of treatment on the original covariates |
| The reported interval omits the adaptive-`g` term | OAT fits the treatment mechanism on `Qbar`, so when `Qbar` is estimated the model class `g` is chosen from is random too, and the influence curve does not see that. The `generated_design` cells measure the consequence: with the design pinned the SE ratio is `properties[generated_design/oracle_design]:se_ratio`, and with it estimated `properties[generated_design/estimated]:se_ratio` |
| Neither design cell's interval on its own excludes 1 | An SE ratio's Monte Carlo error is dominated by the empirical spread in its denominator, worth about two percent at these replication counts. The *paired* difference resolves, because the two cells share their draws and that common error cancels. It runs from `properties[generated_design/estimated]:se_ratio_deficit_lower` to `properties[generated_design/estimated]:se_ratio_deficit_upper`, entirely below zero. The omission is worth a few percent of a reported standard error and does not show up as invalid coverage |
| The design control's margin is a floor on a defect, not a tolerance | If the reported covariance is ever made to carry this term, the control stops being discriminated and this row goes red. That is the correct signal that the limitation has gone stale, rather than a regression |
| The cross-fit overfitting cells are relative evidence | A fully grown tree on this law carries `properties[crossfit_overfitting/cross_fitted_oat]:standardized_bias` empirical standard deviations of nuisance bias. The cell is gated on its SE ratio and on the paired gain rather than on the coverage floor. The primary GLM study carries the absolute gate |
| The parity claim is narrow | It is binary, two-arm, complete-outcome, GLM, and non-cross-fitted. The archived stack fails the analogous continuous law because its length-two outcome bounds enter a scalar `if` condition; the runner treats that as a reference limitation and not a dropped replication. The row does not establish continuous or multi-arm parity, missing outcomes, weights, clusters, strata, simultaneous or bootstrap intervals, broad learner libraries, or severe practical-positivity behaviour. Cross-fitted public behaviour rests on the property study |

A nonparametric bootstrap reruns the whole construction and so carries the omitted terms. It was
measured on this law and did not improve calibration over the reported interval, so it is not
presented here as a remedy.

### Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/README.md)
and [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/manifest.json)
record the provenance and the regeneration commands. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/equivalence.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/properties.csv)
carry every published row.

## Ordinary end-of-study longitudinal TMLE

This study validates `cleverly`'s ordinary, non-cross-fitted two-time-point regimen mean. The law
has monotone censoring and includes static and dynamic plans. The canonical comparison uses R
[`ltmle`](https://www.jstatsoft.org/article/view/v081i01) 1.3-0. The parameter, longitudinal
double robustness, and efficient influence curve follow Bang and Robins (2005), van der Laan and
Gruber (2012), and Petersen et al. (2014).

Agreement with R is secondary to the finite-support functional and Gateaux EIF in
[`tests/discrete_law_longitudinal.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/discrete_law_longitudinal.py).

### What was compared

| setting | `cleverly` | R `ltmle` |
| --- | --- | --- |
| datasets | 1,600 censored samples generated in Python | the identical rows |
| plans | never treat, always treat, and "treat, then continue if L2 is positive" | the same three, invoked once per regimen |
| contrasts | always-minus-never and dynamic-minus-never | the same, from the difference of the two rowwise influence curves so covariance is preserved |
| mechanisms | the generating treatment and censoring probabilities | the same |
| sequential regressions | follower-stratified quasibinomial | the same |
| cumulative-g bounds | nonbinding | nonbinding |
| intervals | pointwise 95% identity-scale Wald, influence-curve variance | the same, `variance.method="ic"` |

The paired discrepancies are numerical-solver scale rather than statistical scale. Each
implementation is also tested against quadrature truth on its own, and both verdicts are carried
beside the paired one. A regeneration fails if either is false.

### Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` | -0.0025 to 0.0015 | 0.9413 | 0.9890 | pass |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `ltmle` | -0.0025 to 0.0015 | 0.9413 | 0.9890 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | `cleverly` | -0.0025 to 0.0016 | 0.9387 | 0.9943 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | R `ltmle` | -0.0025 to 0.0016 | 0.9387 | 0.9943 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` | -0.0015 to 0.000862 | 0.9506 | 1.0130 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `ltmle` | -0.0015 to 0.000862 | 0.9506 | 1.0130 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` | -0.0015 to 0.0018 | 0.9450 | 0.9710 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `ltmle` | -0.0015 to 0.0018 | 0.9450 | 0.9710 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | `cleverly` | -0.0016 to 0.0010 | 0.9469 | 0.9854 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | R `ltmle` | -0.0016 to 0.0010 | 0.9469 | 0.9854 | pass |
<!-- /generated -->

### Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | 2.101e-10 | 4.449e-08 | 1.0000 | 0 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | 6.175e-12 | 1.283e-09 | 1.0000 | 0 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | 8.850e-11 | 3.180e-08 | 1.0000 | 0 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | -1.216e-10 | 3.133e-08 | 1.0000 | 0 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | -1.155e-10 | 3.766e-08 | 1.0000 | 0 | pass |
<!-- /generated -->

### Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `dynamic__both_correct` | positive | dynamic plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0019 to 0.0017, margin 0.0062 | pass |
| `double_robustness` | `dynamic__both_wrong` | control | dynamic plan: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias 0.0146 to 0.0188, margin 0.0070 | pass |
| `double_robustness` | `dynamic__mechanism_correct` | positive | dynamic plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0016 to 0.0022, margin 0.0064 | pass |
| `double_robustness` | `dynamic__outcome_correct` | positive | dynamic plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0020 to 0.0017, margin 0.0062 | pass |
| `double_robustness` | `static__both_correct` | positive | static plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0055 to 0.0030, margin 0.0142 | pass |
| `double_robustness` | `static__both_wrong` | control | static plan: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.0288 to -0.0201, margin 0.0146 | pass |
| `double_robustness` | `static__mechanism_correct` | positive | static plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0036 to 0.0053, margin 0.0149 | pass |
| `double_robustness` | `static__outcome_correct` | positive | static plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0040 to 0.0049, margin 0.0151 | pass |
| `interval_calibration` | `dynamic__correctly_specified` | positive | dynamic plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9282 to 0.9533, SE ratio 0.9552 to 1.0280, empirical efficiency ratio 0.9729 to 1.0461, reported efficiency ratio 0.9967 to 1.0025 | pass |
| `interval_calibration` | `dynamic__noise_control` | control | dynamic plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8133 to 0.8529, SE ratio 0.6883 to 0.7404, empirical efficiency ratio 1.3502 to 1.4513, reported efficiency ratio 0.9967 to 1.0025 | pass |
| `interval_calibration` | `dynamic__shrunken_se_control` | control | dynamic plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8068 to 0.8469, SE ratio 0.6689 to 0.7186, empirical efficiency ratio 0.9737 to 1.0457, reported efficiency ratio 0.6976 to 0.7018 | pass |
| `interval_calibration` | `static__correctly_specified` | positive | static plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9397 to 0.9626, SE ratio 0.9671 to 1.0399, empirical efficiency ratio 0.9591 to 1.0307, reported efficiency ratio 0.9942 to 1.0001 | pass |
| `interval_calibration` | `static__noise_control` | control | static plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8168 to 0.8560, SE ratio 0.6838 to 0.7375, empirical efficiency ratio 1.3532 to 1.4573, reported efficiency ratio 0.9943 to 1.0000 | pass |
| `interval_calibration` | `static__shrunken_se_control` | control | static plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8068 to 0.8469, SE ratio 0.6779 to 0.7281, empirical efficiency ratio 0.9590 to 1.0298, reported efficiency ratio 0.6960 to 0.7000 | pass |
| `power` | `static__alternative` | positive | static plan: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9700, 0.9508 to 0.9833 | pass |
| `root_n_and_efficiency` | `dynamic__n_2000` | positive | dynamic plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000406, coverage 0.9282 to 0.9688, SE ratio 0.9811 | pass |
| `root_n_and_efficiency` | `dynamic__n_500` | control | dynamic plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0020, coverage 0.8822 to 0.9353, SE ratio 0.8963 | pass |
| `root_n_and_efficiency` | `dynamic__n_8000` | positive | dynamic plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000337, coverage 0.9326 to 0.9717, SE ratio 1.0063 | pass |
| `root_n_and_efficiency` | `static__n_2000` | positive | static plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000474, coverage 0.9020 to 0.9502, SE ratio 0.9707 | pass |
| `root_n_and_efficiency` | `static__n_500` | control | static plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0011, coverage 0.8682 to 0.9244, SE ratio 0.8792 | pass |
| `root_n_and_efficiency` | `static__n_8000` | positive | static plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000085, coverage 0.9208 to 0.9637, SE ratio 0.9884 | pass |
| `root_n_rate` | `dynamic__empirical_sd` | positive | dynamic plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5581 to -0.4904 | pass |
| `root_n_rate` | `dynamic__reported_se` | positive | dynamic plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4857 to -0.4776 | pass |
| `root_n_rate` | `static__empirical_sd` | positive | static plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5482 to -0.4841 | pass |
| `root_n_rate` | `static__reported_se` | positive | static plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4786 to -0.4694 | pass |
| `targeting_necessity` | `dynamic__targeted` | positive | dynamic plan: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0014 to 0.0023, margin 0.0063 | pass |
| `targeting_necessity` | `dynamic__untargeted` | control | dynamic plan: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias 0.0239 to 0.0282, margin 0.0073 | pass |
| `targeting_necessity` | `static__targeted` | positive | static plan: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0018 to 0.0068, margin 0.0143 | pass |
| `targeting_necessity` | `static__untargeted` | control | static plan: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias -0.0242 to -0.0156, margin 0.0144 | pass |
| `type_i_error` | `static__sharp_null` | positive | static plan: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0700, 0.0488 to 0.0965 | pass |
<!-- /generated -->

The property study samples the exact binary support law rather than the continuous comparison law.
Its longhand functional supplies exact static and dynamic truths, and its Gateaux derivative
supplies the efficiency bound without reading either estimator.

### Measured values

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
| `property_cells_total` | 30 | independent property cells |
| `property_cells_passed` | 30 | property cells passing |
| `max_standardized_bias` | 0.0180 | largest primary standardized bias |
| `min_coverage` | 0.9387 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9216 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9290 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0641 | highest primary SE-ratio endpoint |
| `max_margin_utilization` | 4.449e-08 | largest share of paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0000 | largest paired RMSE-ratio bound |
| `min_coverage_difference_lower` | 0 | smallest paired coverage-difference bound |
| `max_calibration_excess_upper` | 1.969e-08 | largest paired calibration-excess bound |
| `properties[double_robustness/static__both_wrong]:standardized_bias` | -0.4195 | static both-wrong control |
| `properties[double_robustness/dynamic__both_wrong]:standardized_bias` | 0.5944 | dynamic both-wrong control |
| `properties[root_n_and_efficiency/static__n_500]:coverage` | 0.8988 | static small-sample control coverage |
| `properties[root_n_and_efficiency/dynamic__n_500]:coverage` | 0.9113 | dynamic small-sample control coverage |
| `properties[interval_calibration/static__correctly_specified]:efficiency_empirical_ratio` | 0.9948 | static empirical spread over exact EIF bound |
| `properties[interval_calibration/dynamic__correctly_specified]:efficiency_empirical_ratio` | 1.0093 | dynamic empirical spread over exact EIF bound |
| `properties[type_i_error/static__sharp_null]:rejection_rate` | 0.0700 | confounded-null rejection rate |
| `properties[type_i_error/static__sharp_null]:rejection_ci_upper` | 0.0965 | one-sided bound the type-I ceiling is checked against |
| `properties[power/static__alternative]:rejection_rate` | 0.9700 | alternative rejection rate |
| `properties[root_n_and_efficiency/static__n_2000]:coverage_ci_lower` | 0.9020 | tightest primary property coverage endpoint |
| `properties[targeting_necessity/static__targeted]:standardized_bias` | 0.0435 | static contrast as the estimator computes it |
| `properties[targeting_necessity/static__untargeted]:standardized_bias` | -0.3447 | the same recursion with no fluctuation |
| `properties[targeting_necessity/dynamic__untargeted]:standardized_bias` | 0.8970 | dynamic contrast with no fluctuation |
| `properties[targeting_necessity/static__targeted]:targeting_displacement` | 0.3909 | least-displaced contrast, in targeted standard deviations |
| `max_targeting_displacement` | 0.0938 | largest final-fluctuation move, in standard errors |
| `median_targeting_displacement` | 0.0117 | median final-fluctuation move, in standard errors |
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
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.2500 | least the fluctuation must move the estimate |

### Limitations

| limitation | what it means for use |
| --- | --- |
| The static paired comparisons do not witness targeting | A plug-in built from the same two regressions with no fluctuation at either node still clears both acceptance gates on `ey_regimen[never]`, `ey_regimen[always]`, and `ate_regimen[always vs never]`. Only the two estimands carrying the dynamic rule separate it. The targeting evidence in this row is the dynamic plan and the `targeting_necessity` cells, not the count of paired comparisons. `tests/e2e/test_ltmle_targeting_slow.py` asserts exactly which estimands survive, so the limitation is measured on every run |
| `initial_estimate` measures the final fluctuation only | The earlier node's regression is of the *already targeted* later node. R's `fit$Q[[1]]` regresses the updated `Q.kplus1` and `cleverly`'s first step does the same. So `max_targeting_displacement` and `median_targeting_displacement` measure the last fluctuation rather than the whole targeting step |
| The reference's truth column is not independent of this codebase | The quadrature truth the R rows are scored against comes from `cleverly.datasets.longitudinal`. What makes it usable is that the quadrature is checked separately, against Monte Carlo and by node refinement, in `tests/unit/test_datasets_longitudinal.py` |
| The test over-rejects mildly under the harder null | The rate is 0.0700 at n=4,000 on 800 replications, about 2.6 Monte Carlo standard errors above nominal, and coverage in the same cell is 0.9300. The one-sided bound of 0.0965 clears the predeclared ceiling of 0.1000, so the cell passes, but with roughly three rejections to spare and a point estimate genuinely above 0.05. The influence-curve standard error is slightly optimistic here, and this is published rather than absorbed |
| The n=500 cells are controls whose inference the row does not claim | Each must *resolve*, placing its exact 99% coverage interval clear of nominal on one side or the other. Below the floor is a published small-sample limitation; at or above it is the estimator turning out to be adequate. Only a straddling interval fails, because it is the one outcome that says nothing |
| Positivity is comfortable throughout | The smallest cumulative mechanism product on the comparison law sits between 0.006 and 0.03, and the property law bounds every conditional into [0.25, 0.75]. No cell here speaks to near-positivity behaviour or to an active bound |
| The row is bounded to ordinary end-of-study estimation | Survival has its own row below. This row excludes competing risks, longitudinal MSMs, observation weights, clustering, simultaneous bands, flexible learning, cross-fitting, active truncation, and R parity for learned mechanisms. Those are different estimators or compositions and require their own studies |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, and conditionally independent censoring. Single-correct-nuisance cells establish
consistency only. Calibrated influence-curve inference is claimed where both nuisance sequences are
correct.

### Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle/README.md)
gives the regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle/manifest.json)
carries the package checksum, R image digest, source commit, formulas, seeds, and artifact hashes.
The [replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle/replicates.csv.gz),
[paired verdicts](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle/equivalence.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle/properties.csv)
carry every published row.

The sharp-null law replaces the outcome probabilities on the cells the contrasted plans traverse
and nothing else, so it shares this law's treatment, censoring, and L2 mechanisms exactly. L2 still
moves the outcome from 0.25 to 0.75, censoring is still informative through it, and the first arm
still matters. A baseline-only standardisation returns -0.0088 rather than the truth, so the null
is one an estimator has to work for. Both contrasts are exactly zero under it. Only the static one
is registered as a cell, because a type-I cell needs a nonzero-effect power control and the dynamic
contrast's power at n=4,000 is near 0.43 against a floor of 0.80.

## Cross-fitted end-of-study longitudinal TMLE

This study validates the five-fold end-of-study recursion. Each outer fold fits and targets a
complete recursion on its training rows. The estimator stitches predictions and influence curves
only on held-out rows.

The canonical comparison uses R [`lmtp`](https://github.com/nt-williams/lmtp) 1.5.4 at commit
`f04a2b4`, which is the maintained package that implements a cross-fitted sequential regression.
R `ltmle`, the comparator the two ordinary longitudinal rows use, has no cross-fitting at all, so
it cannot witness this construction.

Agreement with R is secondary to the finite-support functional and Gateaux EIF in
[`tests/discrete_law_longitudinal.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/discrete_law_longitudinal.py).

### What was compared

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
and understates the influence curve: coverage ran from 0.75 to 0.91 and the SE ratio from 0.60 to
0.86, against 0.949 to 0.957 and 1.00 to 1.02 for `cleverly` on the same panels. The sequential
regressions stay misspecified on both sides, so targeting still has work to do.

The exact-law structural test fixes the fold assignment and repeats every support point in each
fold. It checks every regimen mean and correlated contrast against the functional and Gateaux EIF.
The leakage tests also require held-out outcome predictions at both recursion nodes.

### Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` cross-fitted LTMLE | -0.0019 to 0.0022 | 0.9431 | 0.9839 | pass |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `lmtp` | -0.0018 to 0.0023 | 0.9406 | 0.9831 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | `cleverly` cross-fitted LTMLE | -0.0014 to 0.0028 | 0.9525 | 0.9986 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | R `lmtp` | -0.0015 to 0.0026 | 0.9506 | 1.0001 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` cross-fitted LTMLE | -0.000844 to 0.0016 | 0.9531 | 1.0156 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `lmtp` | -0.000873 to 0.0015 | 0.9519 | 1.0171 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` cross-fitted LTMLE | -0.0014 to 0.0018 | 0.9506 | 0.9998 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `lmtp` | -0.0015 to 0.0017 | 0.9469 | 1.0012 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | `cleverly` cross-fitted LTMLE | -0.000445 to 0.0022 | 0.9531 | 1.0028 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | R `lmtp` | -0.000648 to 0.0019 | 0.9506 | 1.0048 | pass |
<!-- /generated -->

### Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | -0.000063 | 0.0132 | 1.0064 | 0.0025 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | 0.000120 | 0.0248 | 1.0131 | 0.0019 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | 0.000034 | 0.0122 | 1.0092 | 0.0012 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | 0.000097 | 0.0256 | 1.0091 | 0.0037 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | 0.000217 | 0.0718 | 1.0199 | 0.0025 | pass |
<!-- /generated -->

### Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `cross_fitted_ltmle` | positive | five-fold end-of-study LTMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 1.1486 to 1.1965 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.3463 to 0.3603 | pass |
| `double_robustness` | `dynamic__both_correct` | positive | dynamic plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0022 to 0.0015, margin 0.0063 | pass |
| `double_robustness` | `dynamic__both_wrong` | control | dynamic plan: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias 0.0165 to 0.0206, margin 0.0069 | pass |
| `double_robustness` | `dynamic__mechanism_correct` | positive | dynamic plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0018 to 0.0022, margin 0.0067 | pass |
| `double_robustness` | `dynamic__outcome_correct` | positive | dynamic plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0021 to 0.0017, margin 0.0063 | pass |
| `double_robustness` | `static__both_correct` | positive | static plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0060 to 0.0026, margin 0.0144 | pass |
| `double_robustness` | `static__both_wrong` | control | static plan: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.0247 to -0.0162, margin 0.0143 | pass |
| `double_robustness` | `static__mechanism_correct` | positive | static plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0028 to 0.0058, margin 0.0145 | pass |
| `double_robustness` | `static__outcome_correct` | positive | static plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0062 to 0.0029, margin 0.0153 | pass |
| `interval_calibration` | `dynamic__correctly_specified` | positive | dynamic plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9388 to 0.9619, SE ratio 0.9832 to 1.0640, empirical efficiency ratio 0.9530 to 1.0304, reported efficiency ratio 1.0106 to 1.0163 | pass |
| `interval_calibration` | `dynamic__noise_control` | control | dynamic plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8098 to 0.8497, SE ratio 0.6952 to 0.7478, empirical efficiency ratio 1.3560 to 1.4571, reported efficiency ratio 1.0106 to 1.0163 | pass |
| `interval_calibration` | `dynamic__shrunken_se_control` | control | dynamic plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8129 to 0.8525, SE ratio 0.6888 to 0.7445, empirical efficiency ratio 0.9533 to 1.0294, reported efficiency ratio 0.7074 to 0.7114 | pass |
| `interval_calibration` | `static__correctly_specified` | positive | static plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9305 to 0.9552, SE ratio 0.9714 to 1.0481, empirical efficiency ratio 0.9740 to 1.0501, reported efficiency ratio 1.0177 to 1.0235 | pass |
| `interval_calibration` | `static__noise_control` | control | static plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8111 to 0.8509, SE ratio 0.6903 to 0.7437, empirical efficiency ratio 1.3730 to 1.4783, reported efficiency ratio 1.0177 to 1.0234 | pass |
| `interval_calibration` | `static__shrunken_se_control` | control | static plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8076 to 0.8477, SE ratio 0.6801 to 0.7337, empirical efficiency ratio 0.9741 to 1.0504, reported efficiency ratio 0.7124 to 0.7164 | pass |
| `power` | `static__alternative` | positive | static plan: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9650, 0.9447 to 0.9796 | pass |
| `root_n_and_efficiency` | `dynamic__n_1000` | control | dynamic plan: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0015, coverage 0.9208 to 0.9637, SE ratio 0.9979 | pass |
| `root_n_and_efficiency` | `dynamic__n_2000` | positive | dynamic plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000376, coverage 0.9223 to 0.9647, SE ratio 0.9995 | pass |
| `root_n_and_efficiency` | `dynamic__n_8000` | positive | dynamic plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000036, coverage 0.9386 to 0.9757, SE ratio 1.0355 | pass |
| `root_n_and_efficiency` | `static__n_1000` | control | static plan: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0037, coverage 0.9121 to 0.9575, SE ratio 1.0067 | pass |
| `root_n_and_efficiency` | `static__n_2000` | positive | static plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0038, coverage 0.9371 to 0.9747, SE ratio 1.0306 | pass |
| `root_n_and_efficiency` | `static__n_8000` | positive | static plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000677, coverage 0.9267 to 0.9677, SE ratio 1.0022 | pass |
| `root_n_rate` | `dynamic__empirical_sd` | positive | dynamic plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5730 to -0.4862 | pass |
| `root_n_rate` | `dynamic__reported_se` | positive | dynamic plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5140 to -0.5074 | pass |
| `root_n_rate` | `static__empirical_sd` | positive | static plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5582 to -0.4697 | pass |
| `root_n_rate` | `static__reported_se` | positive | static plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5211 to -0.5144 | pass |
| `targeting_necessity` | `dynamic__targeted` | positive | dynamic plan: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0020 to 0.0018, margin 0.0065 | pass |
| `targeting_necessity` | `dynamic__untargeted` | control | dynamic plan: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias 0.0232 to 0.0274, margin 0.0070 | pass |
| `targeting_necessity` | `static__targeted` | positive | static plan: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0042 to 0.0050, margin 0.0153 | pass |
| `targeting_necessity` | `static__untargeted` | control | static plan: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias -0.0263 to -0.0173, margin 0.0151 | pass |
| `type_i_error` | `static__sharp_null` | positive | static plan: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0425, 0.0263 to 0.0644 | pass |
<!-- /generated -->

The property study preserves the ordinary row's double-robustness, rate, efficiency, calibration,
null, power, and targeting instruments. It runs each positive estimator with five outer folds.

The overfitting pair uses identical nonlinear panels and a fully grown outcome tree. The positive
arm predicts held-out rows. The control predicts its training rows. The joint verdict requires an
SE ratio inside the declared band and a predeclared coverage gain over the control.

Read the direction of that ratio, not only the verdict. In-sample fitting understates the standard
error by a factor near three, at 0.3532. Cross-fitting removes the understatement and overshoots
it, at 1.1725. A noisy outcome model inflates the residual term of the influence curve, so a
conservative ratio is the expected direction here rather than an anomaly. The cell establishes
that cross-fitting restores honest inference. It does not establish calibration under a fully
grown tree, and the `interval_calibration` family, which does make a calibration claim, uses
correctly specified nuisances instead.

The gate is close. The 99% interval reaches 1.1965 against a ceiling of 1.2000, so a change of
learner, sample size, or fold count could move this cell across it. The replication budget is
8,000 rather than the shared 400 because the shared budget leaves the interval wider than the
remaining margin. That is legitimate for an equivalence-shaped gate, which more replications make
easier rather than harder, and it is recorded here so the margin cannot be read as comfortable.

### Measured values

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
| `max_standardized_bias` | 0.0425 | largest primary standardized bias |
| `min_coverage` | 0.9406 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9237 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9421 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0650 | highest primary SE-ratio endpoint |
| `properties[crossfit_overfitting/cross_fitted_ltmle]:coverage` | 0.9754 | cross-fitted tree coverage |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.5081 | in-sample tree coverage |
| `properties[crossfit_overfitting/cross_fitted_ltmle]:coverage_gain_ci_lower` | 0.4526 | lower bound for the paired coverage gain |
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
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.2500 | least the fluctuation must move the estimate |
| `margin:overfit_se_floor` | 0.8500 | cross-fitted tree SE-ratio lower bound |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample tree SE-ratio upper bound |
| `margin:overfit_coverage_gain` | 0.1500 | minimum paired coverage gain |

### Limitations

| limitation | what it means for use |
| --- | --- |
| Agreement with `lmtp` is distributional, not numerical | The paired claim is that the mean difference sits inside the similarity margin and that `cleverly` is no worse. The two ordinary rows agree with R `ltmle` to solver precision because both run the identical regression; here the sequential regressions are still fitted differently, so per-replication estimates differ at statistical scale |
| One fixed five-fold assignment is studied | The row does not validate repeated folds or time-respecting splits |
| The cross-fit overfitting cell passes near its ceiling | Its 99% SE-ratio interval reaches 1.1965 against a ceiling of 1.2000. The cell shows that cross-fitting restores honest inference under a fully grown tree. It does not show calibration under one |
| Flexible learning is an independent property instrument | The paired comparison uses one GLM learner on each side. The tree pair validates held-out prediction behavior, not parity for learner-library selection |
| The row reports one terminal mean per plan | Survival curves, competing risks, and longitudinal MSM projections have different parameters |
| Inference is pointwise | The row does not validate simultaneous bands, bootstrap intervals, weights, or clustering |
| The mechanism is supplied rather than estimated | Both implementations receive the generating probabilities, so the paired row says nothing about mechanism estimation or about a severe practical-positivity violation. The property study's double-robustness cells cover misspecified mechanisms separately |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, and conditionally independent censoring. Single-correct-nuisance cells establish
consistency only. Calibrated inference uses the cells where both nuisance sequences are correct.

### Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle/README.md)
gives the regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle/manifest.json)
records the seeds, the configuration, the pinned `lmtp` version and source commit, the digest of
every study module and reference source, and the artifact hashes.

## Ordinary survival-curve longitudinal TMLE

This study validates ordinary, non-cross-fitted cumulative risk estimation under absorbing
failure. It covers two horizons, monotone censoring, static plans, and a dynamic plan. The
canonical comparison uses R [`ltmle`](https://www.jstatsoft.org/article/view/v081i01) 1.3-0 with
`survivalOutcome=TRUE`.

The study reports each unique parameter once. At the first horizon, the dynamic plan equals the
always-treated plan by construction. A structural test checks that identity instead of counting
the duplicate as repeated-sampling evidence.

### What was compared

| setting | `cleverly` | R `ltmle` |
| --- | --- | --- |
| datasets | 1,600 censored survival panels generated in Python | the identical rows |
| horizons | cumulative risk by times one and two | one `survivalOutcome=TRUE` fit for each data prefix |
| plans | never treat, always treat, and continue after initial treatment when L2 is positive | the same unique plan and horizon combinations |
| contrasts | always-minus-never at both horizons and dynamic-minus-never at time two | differences of rowwise influence curves, preserving covariance |
| mechanisms | the generating treatment and censoring probabilities | the same fixed probabilities |
| sequential regressions | follower-stratified quasibinomial | the same formulas and link |
| cumulative-g bounds | nonbinding | nonbinding |
| intervals | pointwise 95% identity-scale Wald, influence-curve variance | the same, with `variance.method="ic"` |

R fits each requested prefix because `ltmle` returns the terminal cumulative risk for that input.
The Python implementation performs one backward pass per horizon. Comparing the prefixes tests the
same horizon-specific parameters without treating a terminal outcome as a survival curve.

### Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | `cleverly` | -0.0022 to 0.000243 | 0.9494 | 1.0064 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | R `ltmle` | -0.0022 to 0.000243 | 0.9494 | 1.0064 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | `cleverly` | -0.0036 to 0.000104 | 0.9406 | 0.9938 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | R `ltmle` | -0.0036 to 0.000104 | 0.9406 | 0.9938 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | `cleverly` | -0.0038 to 0.000020 | 0.9387 | 0.9923 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | R `ltmle` | -0.0038 to 0.000020 | 0.9387 | 0.9923 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | `cleverly` | -0.0012 to 0.000324 | 0.9444 | 1.0076 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | R `ltmle` | -0.0012 to 0.000324 | 0.9444 | 1.0076 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | `cleverly` | -0.0018 to 0.000412 | 0.9544 | 1.0105 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | R `ltmle` | -0.0018 to 0.000412 | 0.9544 | 1.0105 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | `cleverly` | -0.000434 to 0.0015 | 0.9519 | 0.9976 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | R `ltmle` | -0.000434 to 0.0015 | 0.9519 | 0.9976 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | `cleverly` | -0.000431 to 0.0025 | 0.9425 | 0.9999 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | R `ltmle` | -0.000431 to 0.0025 | 0.9425 | 0.9999 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | `cleverly` | -0.0020 to 0.000348 | 0.9537 | 1.0104 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | R `ltmle` | -0.0020 to 0.000348 | 0.9537 | 1.0104 | pass |
<!-- /generated -->

### Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | 3.324e-11 | 1.160e-08 | 1.0000 | 0 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | -3.360e-10 | 7.845e-08 | 1.0000 | 0 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | -3.903e-10 | 8.898e-08 | 1.0000 | 0 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | -8.728e-11 | 4.910e-08 | 1.0000 | 0 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | -2.473e-10 | 9.571e-08 | 1.0000 | 0 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | -1.205e-10 | 5.271e-08 | 1.0000 | 0 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | 8.867e-11 | 2.594e-08 | 1.0000 | 0 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | -3.016e-10 | 1.104e-07 | 1.0000 | 0 | pass |
<!-- /generated -->

### Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `dynamic_t2__both_correct` | positive | dynamic plan at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0023 to 0.0042, margin 0.0109 | pass |
| `double_robustness` | `dynamic_t2__both_wrong` | control | dynamic plan at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.0447 to -0.0382, margin 0.0109 | pass |
| `double_robustness` | `dynamic_t2__mechanism_correct` | positive | dynamic plan at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0029 to 0.0037, margin 0.0111 | pass |
| `double_robustness` | `dynamic_t2__outcome_correct` | positive | dynamic plan at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0010 to 0.0055, margin 0.0109 | pass |
| `double_robustness` | `static_t1__both_correct` | positive | static plan at horizon one: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0022 to 0.0018, margin 0.0067 | pass |
| `double_robustness` | `static_t1__both_wrong` | control | static plan at horizon one: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.0449 to -0.0408, margin 0.0068 | pass |
| `double_robustness` | `static_t1__mechanism_correct` | positive | static plan at horizon one: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0024 to 0.0016, margin 0.0068 | pass |
| `double_robustness` | `static_t1__outcome_correct` | positive | static plan at horizon one: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.000895 to 0.0033, margin 0.0071 | pass |
| `double_robustness` | `static_t2__both_correct` | positive | static plan at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0025 to 0.0043, margin 0.0114 | pass |
| `double_robustness` | `static_t2__both_wrong` | control | static plan at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.0368 to -0.0301, margin 0.0113 | pass |
| `double_robustness` | `static_t2__mechanism_correct` | positive | static plan at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0026 to 0.0042, margin 0.0113 | pass |
| `double_robustness` | `static_t2__outcome_correct` | positive | static plan at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.000743 to 0.0059, margin 0.0112 | pass |
| `interval_calibration` | `dynamic_t2__correctly_specified` | positive | dynamic plan at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9300 to 0.9429, SE ratio 0.9483 to 0.9849, empirical efficiency ratio 1.0094 to 1.0482, reported efficiency ratio 0.9922 to 0.9961 | pass |
| `interval_calibration` | `dynamic_t2__noise_control` | control | dynamic plan at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8113 to 0.8315, SE ratio 0.6760 to 0.7028, empirical efficiency ratio 1.4154 to 1.4707, reported efficiency ratio 0.9922 to 0.9961 | pass |
| `interval_calibration` | `dynamic_t2__shrunken_se_control` | control | dynamic plan at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8042 to 0.8247, SE ratio 0.6636 to 0.6894, empirical efficiency ratio 1.0096 to 1.0484, reported efficiency ratio 0.6945 to 0.6972 | pass |
| `interval_calibration` | `static_t1__correctly_specified` | positive | static plan at horizon one: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9393 to 0.9513, SE ratio 0.9700 to 1.0068, empirical efficiency ratio 0.9930 to 1.0306, reported efficiency ratio 0.9993 to 1.0003 | pass |
| `interval_calibration` | `static_t1__noise_control` | control | static plan at horizon one: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8175 to 0.8374, SE ratio 0.6827 to 0.7091, empirical efficiency ratio 1.4098 to 1.4644, reported efficiency ratio 0.9993 to 1.0003 | pass |
| `interval_calibration` | `static_t1__shrunken_se_control` | control | static plan at horizon one: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8177 to 0.8376, SE ratio 0.6788 to 0.7047, empirical efficiency ratio 0.9932 to 1.0310, reported efficiency ratio 0.6995 to 0.7002 | pass |
| `interval_calibration` | `static_t2__correctly_specified` | positive | static plan at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9276 to 0.9407, SE ratio 0.9411 to 0.9774, empirical efficiency ratio 1.0122 to 1.0515, reported efficiency ratio 0.9874 to 0.9914 | pass |
| `interval_calibration` | `static_t2__noise_control` | control | static plan at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8060 to 0.8264, SE ratio 0.6757 to 0.7008, empirical efficiency ratio 1.4119 to 1.4641, reported efficiency ratio 0.9874 to 0.9915 | pass |
| `interval_calibration` | `static_t2__shrunken_se_control` | control | static plan at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8001 to 0.8208, SE ratio 0.6585 to 0.6842, empirical efficiency ratio 1.0125 to 1.0519, reported efficiency ratio 0.6911 to 0.6941 | pass |
| `power` | `dynamic_t2__alternative` | positive | dynamic plan at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `power` | `static_t1__alternative` | positive | static plan at horizon one: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `power` | `static_t2__alternative` | positive | static plan at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `dynamic_t2__n_1000` | control | dynamic plan at horizon two: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias 0.0020, coverage 0.8766 to 0.9309, SE ratio 0.8690 | pass |
| `root_n_and_efficiency` | `dynamic_t2__n_2000` | positive | dynamic plan at horizon two: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0016, coverage 0.9136 to 0.9585, SE ratio 0.9810 | pass |
| `root_n_and_efficiency` | `dynamic_t2__n_8000` | positive | dynamic plan at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000974, coverage 0.9150 to 0.9596, SE ratio 0.9872 | pass |
| `root_n_and_efficiency` | `static_t1__n_1000` | control | static plan at horizon one: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias 0.000396, coverage 0.9107 to 0.9565, SE ratio 0.9596 | pass |
| `root_n_and_efficiency` | `static_t1__n_2000` | positive | static plan at horizon one: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0012, coverage 0.9341 to 0.9727, SE ratio 1.0450 | pass |
| `root_n_and_efficiency` | `static_t1__n_8000` | positive | static plan at horizon one: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000241, coverage 0.9179 to 0.9616, SE ratio 0.9838 | pass |
| `root_n_and_efficiency` | `static_t2__n_1000` | control | static plan at horizon two: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.000599, coverage 0.8696 to 0.9255, SE ratio 0.8669 | pass |
| `root_n_and_efficiency` | `static_t2__n_2000` | positive | static plan at horizon two: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0024, coverage 0.9208 to 0.9637, SE ratio 0.9893 | pass |
| `root_n_and_efficiency` | `static_t2__n_8000` | positive | static plan at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000698, coverage 0.9121 to 0.9575, SE ratio 0.9733 | pass |
| `root_n_rate` | `dynamic_t2__empirical_sd` | positive | dynamic plan at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5866 to -0.4928 | pass |
| `root_n_rate` | `dynamic_t2__reported_se` | positive | dynamic plan at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4906 to -0.4814 | pass |
| `root_n_rate` | `static_t1__empirical_sd` | positive | static plan at horizon one: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5463 to -0.4630 | pass |
| `root_n_rate` | `static_t1__reported_se` | positive | static plan at horizon one: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5008 to -0.4986 | pass |
| `root_n_rate` | `static_t2__empirical_sd` | positive | static plan at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5701 to -0.4752 | pass |
| `root_n_rate` | `static_t2__reported_se` | positive | static plan at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4811 to -0.4709 | pass |
| `survival_recursion_necessity` | `always_t2__survival` | positive | always-treat risk at horizon two: the survival estimator keeps failures in their event node and removes them afterward | bias interval inside the equivalence margin | bias -0.0038 to 0.0022, margin 0.0100 | pass |
| `survival_recursion_necessity` | `always_t2__survivor_only` | control | always-treat risk at horizon two: the same horizon-two outcome analyzed only among first-node survivors | bias interval must fall entirely outside the margin | bias -0.1523 to -0.1430, margin 0.0155 | pass |
| `targeting_necessity` | `dynamic_t2__targeted` | positive | dynamic plan at horizon two: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0035 to 0.0029, margin 0.0108 | pass |
| `targeting_necessity` | `dynamic_t2__untargeted` | control | dynamic plan at horizon two: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias -0.0444 to -0.0379, margin 0.0108 | pass |
| `targeting_necessity` | `static_t1__targeted` | positive | static plan at horizon one: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0019 to 0.0021, margin 0.0068 | pass |
| `targeting_necessity` | `static_t1__untargeted` | control | static plan at horizon one: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias -0.0448 to -0.0408, margin 0.0067 | pass |
| `targeting_necessity` | `static_t2__targeted` | positive | static plan at horizon two: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0033 to 0.0034, margin 0.0113 | pass |
| `targeting_necessity` | `static_t2__untargeted` | control | static plan at horizon two: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias -0.0366 to -0.0299, margin 0.0114 | pass |
| `type_i_error` | `dynamic_t2__sharp_null` | positive | dynamic plan at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0537, 0.0353 to 0.0777 | pass |
| `type_i_error` | `static_t1__sharp_null` | positive | static plan at horizon one: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0500, 0.0323 to 0.0733 | pass |
| `type_i_error` | `static_t2__sharp_null` | positive | static plan at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0575, 0.0384 to 0.0821 | pass |
<!-- /generated -->

The property study samples an exact binary survival law. It checks static risks at both horizons
and the dynamic risk at time two. Its independent instruments include double robustness, exact-EIF
efficiency, root-n rates, calibration, null size, power, and targeting controls.

The survival-recursion control is separate from targeting. It replaces cumulative risk with a
survivor-only terminal hazard while leaving the rest of the procedure unchanged. The control must
miss the exact time-two risk, while the correct recursion must recover it.

The two horizons do not behave alike. The study measures that difference on one set of draws, so
the comparison is internal to the calibration cell rather than across studies.

The first horizon is calibrated. Its SE ratio and its coverage both surround their nominal values.
At the second horizon the whole SE-ratio interval sits below one and the whole coverage interval
sits below nominal. The empirical spread runs above the exact efficiency bound there, while the
reported standard error sits on it. So the reported standard error understates the sampling spread
at the second horizon by a few percent. The measured-values table below carries all four endpoints.

Both intervals stay inside the declared calibration bands, so the cell passes. The second backward
regression is the source, and it is why this cell carries four times the shared calibration budget.
A reader who needs calibrated horizon-two inference at this sample size should treat the shortfall
as measured rather than as absent.

The sharp null keeps the treatment, censoring, and `L2` mechanisms of the law and replaces the two
hazards. Every replacement value is a multiple of one quarter, so an `N`-row sample realises the
null law exactly, as it does the law it derives from. All three contrasts are exactly zero under
it.

At the second horizon a baseline-only standardisation returns 0.0349 rather than zero, so the null
is one an estimator has to be longitudinal to find. At the first horizon it returns exactly zero,
because no time-varying node precedes the first event node. The first-horizon type-I cells
therefore test baseline and censoring adjustment and not longitudinal adjustment. A crude
comparison of arms is biased by -0.05 under the null at that horizon, so those cells are not
vacuous either.

The power alternative keeps the law's own first hazard and replaces the second. It gives the two
horizon-two contrasts different values, so the two horizon-two power cells report two parameters
rather than one number twice.

### Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_total` | 16 | implementation-estimand truth tests |
| `independent_tests_passed` | 16 | truth tests passing |
| `paired_tests_total` | 8 | paired estimand comparisons |
| `paired_tests_passed` | 8 | paired comparisons passing |
| `property_cells_total` | 50 | independent property cells |
| `property_cells_passed` | 50 | property cells passing |
| `max_standardized_bias` | 0.0638 | largest primary standardized bias |
| `min_coverage` | 0.9387 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9216 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9491 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0615 | highest primary SE-ratio endpoint |
| `max_margin_utilization` | 1.104e-07 | largest share of paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0000 | largest paired RMSE-ratio bound |
| `min_coverage_difference_lower` | 0 | smallest paired coverage-difference bound |
| `max_calibration_excess_upper` | 8.055e-08 | largest paired calibration-excess bound |
| `max_targeting_displacement` | 0.3560 | largest final-fluctuation move, in standard errors |
| `median_targeting_displacement` | 0.0343 | median final-fluctuation move, in standard errors |
| `properties[targeting_necessity/static_t1__targeted]:targeting_displacement` | 0.7387 | least-displaced contrast, in targeted standard deviations |
| `properties[survival_recursion_necessity/always_t2__survival]:recursion_displacement` | 3.6840 | survivor-only control's distance, in recursion standard deviations |
| `properties[interval_calibration/static_t1__correctly_specified]:se_ratio` | 0.9880 | horizon-one reported SE over empirical spread |
| `properties[interval_calibration/static_t1__correctly_specified]:coverage` | 0.9455 | horizon-one calibration coverage |
| `properties[interval_calibration/static_t2__correctly_specified]:se_ratio_ci_upper` | 0.9774 | highest horizon-two SE ratio the draws support |
| `properties[interval_calibration/static_t2__correctly_specified]:coverage_ci_upper` | 0.9407 | highest horizon-two coverage the draws support |
| `properties[interval_calibration/static_t2__correctly_specified]:replicates` | 9600 | replications the horizon-two calibration cell required |
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
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.2500 | least the fluctuation must move the estimate |
| `margin:recursion_displacement` | 0.2500 | least the survivor-only control must move the estimate |

### Limitations

| limitation | what it means for use |
| --- | --- |
| This is the ordinary row | Nuisances are fitted on the analysis sample. The separate cross-fitted row below validates held-out nuisance fitting and fold-specific recursion |
| Inference is pointwise | The study reports each horizon separately. It does not validate a simultaneous confidence band for the whole curve. Each horizon is also targeted in its own backward pass, so the reported curve is not constrained to increase |
| Horizon-two inference is mildly anticonservative at n = 2,000 | The reported standard error sits a few percent below the sampling spread, and coverage sits below nominal. Both endpoints stay inside the declared calibration bands. The first horizon does not show this on the same draws |
| The event process has two horizons | Both backward prefixes are exercised. Longer follow-up may compound the horizon-two shortfall above, and this study cannot say by how much |
| The first-horizon null is not a longitudinal null | No time-varying node precedes the first event node, so a baseline-only standardisation recovers that null exactly. The first-horizon type-I cells test baseline and censoring adjustment only |
| `initial_estimate` measures the final fluctuation only | The earlier node's regression is of the *already targeted* later node. R's `fit$Q[[1]]` regresses the updated `Q.kplus1` and `cleverly`'s first step does the same. So `max_targeting_displacement` and `median_targeting_displacement` measure the last fluctuation rather than the whole targeting step. They are not the quantity `margin:targeting_displacement` bounds, which is the targeted-against-unfluctuated distance in the property study |
| The paired study fixes the mechanisms | It isolates survival recursion, sequential regression, targeting, and inference. It does not claim parity for learned treatment or censoring models |
| Positivity is comfortable | No primary cell validates active truncation or near-positivity behavior |
| The size ladder starts at n = 1,000 | Every other registered study starts at 500. An absorbing event thins the risk set, so at 500 some samples of the property law leave the horizon-two parameter unestimable and the estimator refuses. The rate cells therefore say nothing below 1,000 |
| Failure is absorbing and has one cause | Competing-risk cumulative incidence remains a different parameter with no R parity claim |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, and conditionally independent censoring. The event process must also be coded so a
failure remains in the cumulative risk at later horizons.

### Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_survival/README.md)
gives the regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_survival/manifest.json)
records the pinned R package, source commit, image digest, formulas, seeds, and artifact hashes.
The [replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_survival/replicates.csv.gz),
[paired verdicts](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_survival/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_survival/properties.csv)
carry every published row.

## Cross-fitted survival-curve longitudinal TMLE

This study validates five-fold cumulative-risk estimation under absorbing failure. It covers two
horizons, monotone censoring, static plans, and a dynamic plan. Each fold fits and targets its
complete backward recursion on training rows before it evaluates held-out rows.

The canonical comparison uses R [`lmtp`](https://github.com/nt-williams/lmtp) 1.5.4 at commit
`f04a2b4`, on the same panels and the same stored fold assignment, with one fitted prefix per
reported horizon. R `ltmle` has no cross-fitting, so it cannot witness this construction.

Agreement with R is secondary to the finite-support functional and Gateaux EIF in
[`tests/discrete_law_survival.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/discrete_law_survival.py).

### What was compared

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets and folds | 1,600 censored survival panels, each with one exact five-fold assignment | the identical rows and the identical assignment |
| horizons | cumulative risk at times one and two | one fitted prefix per horizon |
| plans | never treat, always treat, and continue after initial treatment when L2 is positive | the same three, as shifted treatment columns |
| contrasts | always-minus-never at both horizons and dynamic-minus-never at time two | the same, from rowwise influence-curve differences |
| treatment and censoring mechanisms | the generating probabilities from the law | the same probabilities, supplied as exact per-node density ratios |
| sequential regressions | quasibinomial GLMs within each outer training set | `SL.glm`, within the same training sets |
| intervals | pointwise 95% identity-scale Wald intervals | the same, after the horizon-two transformation below |

Both implementations receive the mechanism from the law rather than estimating it. `lmtp` has no
`gform` argument, so the adapter substitutes exact per-node density ratios into it, the same way
it substitutes the fold assignment. The substitution is checked on every run against `lmtp`'s own
estimate: the zero pattern must agree cell for cell, and the cumulative ratio must track it.

That choice is what makes the comparison a comparison. An earlier version let each side estimate
the mechanism its own way, and the result measured two unrelated pipelines. `lmtp` fits its ratio
with `SL.glm`, whose linear logit cannot represent the exact classifier log-odds `-log g` for a
deterministic regime, so the ratio came out shrunken. A shrunken clever covariate under-targets
and understates the influence curve: coverage ran from 0.75 to 0.91 and the SE ratio from 0.60 to
0.86, against 0.949 to 0.957 and 1.00 to 1.02 for `cleverly` on the same panels. The sequential
regressions stay misspecified on both sides, so targeting still has work to do.

The retained R runner uses the one-node binary mean at horizon one because `lmtp` requires two
event nodes for its survival path. At horizon two, it converts event-free survival to cumulative
risk and reverses the influence-curve sign. The registered run invokes that runner, so the
transformation is exercised on every regeneration rather than described.

The exact-law structural test repeats every support point in each fold. It checks every unique risk
and contrast against the finite functional and Gateaux EIF. Separate mutations check held-out
outcome predictions and the first-horizon risk-set boundary.

### Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | `cleverly` cross-fitted survival LTMLE | -0.000326 to 0.0022 | 0.9513 | 0.9852 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | R `lmtp` | -0.000339 to 0.0022 | 0.9513 | 0.9849 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | `cleverly` cross-fitted survival LTMLE | -0.0013 to 0.0024 | 0.9469 | 0.9964 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.0012 to 0.0025 | 0.9431 | 0.9953 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | `cleverly` cross-fitted survival LTMLE | -0.0012 to 0.0026 | 0.9444 | 0.9908 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.0011 to 0.0027 | 0.9456 | 0.9890 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | `cleverly` cross-fitted survival LTMLE | -0.000525 to 0.0010 | 0.9537 | 0.9993 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | R `lmtp` | -0.000503 to 0.0010 | 0.9563 | 0.9984 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | `cleverly` cross-fitted survival LTMLE | -0.000904 to 0.0014 | 0.9506 | 0.9912 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | R `lmtp` | -0.000913 to 0.0014 | 0.9513 | 0.9918 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | `cleverly` cross-fitted survival LTMLE | -0.0017 to 0.000295 | 0.9469 | 1.0001 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | R `lmtp` | -0.0016 to 0.000329 | 0.9481 | 1.0003 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | `cleverly` cross-fitted survival LTMLE | -0.0018 to 0.0012 | 0.9463 | 1.0129 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | R `lmtp` | -0.0019 to 0.0011 | 0.9525 | 1.0120 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | `cleverly` cross-fitted survival LTMLE | -0.000777 to 0.0016 | 0.9500 | 0.9877 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | R `lmtp` | -0.000776 to 0.0016 | 0.9469 | 0.9874 | pass |
<!-- /generated -->

### Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | 0.000013 | 0.0044 | 1.0014 | 0 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | -0.000083 | 0.0193 | 1.0060 | 0.0037 | pass |
| two-time-point absorbing-event law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | -0.000094 | 0.0211 | 1.0055 | -0.0013 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | -0.000022 | 0.0121 | 1.0014 | -0.0025 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | 0.000010 | 0.0038 | 1.0040 | -0.000625 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | -0.000035 | 0.0152 | 1.0026 | -0.0012 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | 0.000093 | 0.0273 | 1.0089 | -0.0062 | pass |
| two-time-point absorbing-event law with monotone censoring | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | -3.147e-07 | 0.000112 | 1.0051 | 0.0031 | pass |
<!-- /generated -->

### Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `cross_fitted_survival_ltmle` | positive | five-fold horizon-two survival LTMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 1.0766 to 1.1219 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.3963 to 0.4128 | pass |
| `double_robustness` | `dynamic_t2__both_correct` | positive | dynamic plan at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0062 to 0.000326, margin 0.0110 | pass |
| `double_robustness` | `dynamic_t2__both_wrong` | control | dynamic plan at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.0454 to -0.0386, margin 0.0113 | pass |
| `double_robustness` | `dynamic_t2__mechanism_correct` | positive | dynamic plan at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0031 to 0.0035, margin 0.0112 | pass |
| `double_robustness` | `dynamic_t2__outcome_correct` | positive | dynamic plan at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0037 to 0.0027, margin 0.0107 | pass |
| `double_robustness` | `static_t1__both_correct` | positive | static plan at horizon one: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0024 to 0.0018, margin 0.0069 | pass |
| `double_robustness` | `static_t1__both_wrong` | control | static plan at horizon one: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.0451 to -0.0411, margin 0.0068 | pass |
| `double_robustness` | `static_t1__mechanism_correct` | positive | static plan at horizon one: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0025 to 0.0018, margin 0.0072 | pass |
| `double_robustness` | `static_t1__outcome_correct` | positive | static plan at horizon one: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0024 to 0.0018, margin 0.0071 | pass |
| `double_robustness` | `static_t2__both_correct` | positive | static plan at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0054 to 0.0012, margin 0.0112 | pass |
| `double_robustness` | `static_t2__both_wrong` | control | static plan at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.0378 to -0.0308, margin 0.0117 | pass |
| `double_robustness` | `static_t2__mechanism_correct` | positive | static plan at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0023 to 0.0045, margin 0.0114 | pass |
| `double_robustness` | `static_t2__outcome_correct` | positive | static plan at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0043 to 0.0023, margin 0.0111 | pass |
| `interval_calibration` | `dynamic_t2__correctly_specified` | positive | dynamic plan at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9409 to 0.9528, SE ratio 0.9901 to 1.0290, empirical efficiency ratio 0.9967 to 1.0349, reported efficiency ratio 1.0233 to 1.0271 | pass |
| `interval_calibration` | `dynamic_t2__noise_control` | control | dynamic plan at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8290 to 0.8484, SE ratio 0.7019 to 0.7287, empirical efficiency ratio 1.4071 to 1.4602, reported efficiency ratio 1.0234 to 1.0271 | pass |
| `interval_calibration` | `dynamic_t2__shrunken_se_control` | control | dynamic plan at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8252 to 0.8448, SE ratio 0.6930 to 0.7205, empirical efficiency ratio 0.9959 to 1.0354, reported efficiency ratio 0.7163 to 0.7190 | pass |
| `interval_calibration` | `static_t1__correctly_specified` | positive | static plan at horizon one: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9450 to 0.9564, SE ratio 0.9830 to 1.0200, empirical efficiency ratio 0.9823 to 1.0192, reported efficiency ratio 1.0014 to 1.0023 | pass |
| `interval_calibration` | `static_t1__noise_control` | control | static plan at horizon one: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8223 to 0.8420, SE ratio 0.6930 to 0.7196, empirical efficiency ratio 1.3920 to 1.4455, reported efficiency ratio 1.0014 to 1.0023 | pass |
| `interval_calibration` | `static_t1__shrunken_se_control` | control | static plan at horizon one: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8195 to 0.8394, SE ratio 0.6885 to 0.7139, empirical efficiency ratio 0.9823 to 1.0187, reported efficiency ratio 0.7010 to 0.7016 | pass |
| `interval_calibration` | `static_t2__correctly_specified` | positive | static plan at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9421 to 0.9539, SE ratio 0.9964 to 1.0351, empirical efficiency ratio 1.0008 to 1.0391, reported efficiency ratio 1.0336 to 1.0376 | pass |
| `interval_calibration` | `static_t2__noise_control` | control | static plan at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8353 to 0.8544, SE ratio 0.7115 to 0.7390, empirical efficiency ratio 1.4018 to 1.4551, reported efficiency ratio 1.0337 to 1.0377 | pass |
| `interval_calibration` | `static_t2__shrunken_se_control` | control | static plan at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8265 to 0.8461, SE ratio 0.6973 to 0.7248, empirical efficiency ratio 1.0004 to 1.0391, reported efficiency ratio 0.7235 to 0.7264 | pass |
| `power` | `dynamic_t2__alternative` | positive | dynamic plan at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `power` | `static_t1__alternative` | positive | static plan at horizon one: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `power` | `static_t2__alternative` | positive | static plan at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `dynamic_t2__n_1000` | control | dynamic plan at horizon two: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0012, coverage 0.9238 to 0.9657, SE ratio 1.0058 | pass |
| `root_n_and_efficiency` | `dynamic_t2__n_2000` | positive | dynamic plan at horizon two: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000093, coverage 0.9297 to 0.9698, SE ratio 1.0184 | pass |
| `root_n_and_efficiency` | `dynamic_t2__n_8000` | positive | dynamic plan at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000210, coverage 0.9267 to 0.9677, SE ratio 1.0023 | pass |
| `root_n_and_efficiency` | `static_t1__n_1000` | control | static plan at horizon one: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias 0.000967, coverage 0.9386 to 0.9757, SE ratio 1.0200 | pass |
| `root_n_and_efficiency` | `static_t1__n_2000` | positive | static plan at horizon one: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000180, coverage 0.9431 to 0.9786, SE ratio 1.0253 | pass |
| `root_n_and_efficiency` | `static_t1__n_8000` | positive | static plan at horizon one: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000046, coverage 0.9252 to 0.9667, SE ratio 0.9787 | pass |
| `root_n_and_efficiency` | `static_t2__n_1000` | control | static plan at horizon two: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0016, coverage 0.9238 to 0.9657, SE ratio 1.0305 | pass |
| `root_n_and_efficiency` | `static_t2__n_2000` | positive | static plan at horizon two: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0010, coverage 0.9356 to 0.9737, SE ratio 1.0431 | pass |
| `root_n_and_efficiency` | `static_t2__n_8000` | positive | static plan at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000205, coverage 0.9311 to 0.9708, SE ratio 1.0154 | pass |
| `root_n_rate` | `dynamic_t2__empirical_sd` | positive | dynamic plan at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5632 to -0.4776 | pass |
| `root_n_rate` | `dynamic_t2__reported_se` | positive | dynamic plan at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5278 to -0.5192 | pass |
| `root_n_rate` | `static_t1__empirical_sd` | positive | static plan at horizon one: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5241 to -0.4340 | pass |
| `root_n_rate` | `static_t1__reported_se` | positive | static plan at horizon one: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5022 to -0.5002 | pass |
| `root_n_rate` | `static_t2__empirical_sd` | positive | static plan at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5632 to -0.4789 | pass |
| `root_n_rate` | `static_t2__reported_se` | positive | static plan at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5343 to -0.5252 | pass |
| `survival_recursion_necessity` | `always_t2__survival` | positive | always-treat risk at horizon two: the survival estimator keeps failures in their event node and removes them afterward | bias interval inside the equivalence margin | bias -0.0027 to 0.0034, margin 0.0103 | pass |
| `survival_recursion_necessity` | `always_t2__survivor_only` | control | always-treat risk at horizon two: the same horizon-two outcome analyzed only among first-node survivors | bias interval must fall entirely outside the margin | bias -0.1510 to -0.1413, margin 0.0163 | pass |
| `targeting_necessity` | `dynamic_t2__targeted` | positive | dynamic plan at horizon two: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0045 to 0.0021, margin 0.0111 | pass |
| `targeting_necessity` | `dynamic_t2__untargeted` | control | dynamic plan at horizon two: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias -0.0456 to -0.0391, margin 0.0109 | pass |
| `targeting_necessity` | `static_t1__targeted` | positive | static plan at horizon one: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0035 to 0.0010, margin 0.0075 | pass |
| `targeting_necessity` | `static_t1__untargeted` | control | static plan at horizon one: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias -0.0461 to -0.0417, margin 0.0073 | pass |
| `targeting_necessity` | `static_t2__targeted` | positive | static plan at horizon two: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0038 to 0.0030, margin 0.0115 | pass |
| `targeting_necessity` | `static_t2__untargeted` | control | static plan at horizon two: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias -0.0372 to -0.0303, margin 0.0115 | pass |
| `type_i_error` | `dynamic_t2__sharp_null` | positive | dynamic plan at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0612, 0.0415 to 0.0864 | pass |
| `type_i_error` | `static_t1__sharp_null` | positive | static plan at horizon one: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0612, 0.0415 to 0.0864 | pass |
| `type_i_error` | `static_t2__sharp_null` | positive | static plan at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0587, 0.0394 to 0.0835 | pass |
<!-- /generated -->

The property study preserves every ordinary survival instrument. These include horizon-specific
double robustness, targeting, root-n, efficiency, calibration, null, power, and survivor-only
recursion controls.

The overfitting pair uses a fully grown outcome tree at horizon two. Both arms use identical
nonlinear survival panels and known treatment and censoring mechanisms. The joint verdict requires
held-out fitting to restore SE scale and improve coverage over the in-sample control.

### Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_total` | 16 | implementation-estimand truth tests |
| `independent_tests_passed` | 16 | truth tests passing |
| `paired_tests_total` | 8 | paired estimand comparisons |
| `paired_tests_passed` | 8 | paired comparisons passing |
| `property_cells_total` | 52 | independent property cells |
| `property_cells_passed` | 52 | property cells passing |
| `max_standardized_bias` | 0.0478 | largest primary standardized bias |
| `min_coverage` | 0.9431 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9265 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9395 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0609 | highest primary SE-ratio endpoint |
| `properties[crossfit_overfitting/cross_fitted_survival_ltmle]:coverage` | 0.9646 | cross-fitted tree coverage |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.5693 | in-sample tree coverage |
| `properties[crossfit_overfitting/cross_fitted_survival_ltmle]:coverage_gain_ci_lower` | 0.3812 | lower bound for the paired coverage gain |
| `properties[crossfit_overfitting/cross_fitted_survival_ltmle]:replicates` | 8000 | paired overfitting replications |
| `properties[survival_recursion_necessity/always_t2__survival]:recursion_displacement` | 3.5625 | survivor-only control displacement |
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
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.2500 | least the fluctuation must move the estimate |
| `margin:recursion_displacement` | 0.2500 | least the survivor-only control must move the estimate |
| `margin:overfit_se_floor` | 0.8500 | cross-fitted tree SE-ratio lower bound |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample tree SE-ratio upper bound |
| `margin:overfit_coverage_gain` | 0.1500 | minimum paired coverage gain |

### Limitations

| limitation | what it means for use |
| --- | --- |
| Agreement with `lmtp` is distributional, not numerical | The paired claim is similarity of means and non-inferiority. Per-replication estimates differ at statistical scale, because only the mechanism is shared and the sequential regressions are still fitted differently |
| One fixed five-fold assignment is studied | The row does not validate repeated folds or time-respecting splits |
| Horizon one uses the binary-mean path in R | `lmtp` requires two event nodes for its survival path. The one-node binary mean is the same first-horizon cumulative-risk parameter |
| Inference is pointwise | The row does not validate a simultaneous curve band or enforce monotone reported estimates |
| The event process has two horizons and one cause | Longer follow-up and competing-risk cumulative incidence need separate evidence |
| Flexible learning is an independent property instrument | The paired comparison uses one GLM learner on each side. The tree pair does not establish parity for learner-library selection |
| The primary mechanism bounds are nonbinding | No primary cell validates active bounds, weights, clustering, or severe practical-positivity violations |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, conditionally independent censoring, and correct absorbing-event coding.

### Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle_survival/README.md)
gives the regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle_survival/manifest.json)
records the seeds, the configuration, the pinned `lmtp` version and source commit, the digest of
every study module and reference source, and the artifact hashes.
