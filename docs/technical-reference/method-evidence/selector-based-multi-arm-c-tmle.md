# Selector-based multi-arm C-TMLE

This reporting study exercises the greedy, ordered, and discrete selector strategies for a
labelled three-arm treatment. No canonical implementation is compared: R `ctmle` 0.1.2 is
binary-treatment only, and separate binary fits would target different parameters rather than
form a legitimate multi-arm comparator.

Each strategy is tested against the known truth for three arm means and two ATEs, risk ratios,
and odds ratios. The property record runs all three selector paths beside a forced empty path on
identical draws. Each path reports its own root-mean-square error against that control. A path
that reaches a ratio of one chose the control's mechanism path, so its pair is not a control.

The primary fit is not cross-fitted, and it still draws a split. The selector scores its
candidate path over five selection folds. That split reads no label here: the fit declares
`stratify_folds="none"`, and it asserts the realized partition before it reports a row. The
property cells declare the same split, and there it reaches the outer folds, the selection folds
and the nested folds of each candidate. No cell declares `q_bounds`, because every law in this
row has a binary outcome, whose outcome scaler is already the identity.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-arm binary-outcome law, discrete selector | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` multi-arm selector C-TMLE | -0.0020 to 0.0036 | 0.9400 | 0.9859 | pass |
| three-arm binary-outcome law, discrete selector | `ate[medium vs high]` | difference in counterfactual means, medium versus high | `cleverly` multi-arm selector C-TMLE | -0.0015 to 0.0040 | 0.9400 | 0.9799 | pass |
| three-arm binary-outcome law, discrete selector | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` multi-arm selector C-TMLE | -0.0027 to 0.0012 | 0.9437 | 0.9882 | pass |
| three-arm binary-outcome law, discrete selector | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` multi-arm selector C-TMLE | -0.0020 to 0.0020 | 0.9487 | 0.9856 | pass |
| three-arm binary-outcome law, discrete selector | `ey[medium]` | counterfactual mean under treatment arm 'medium' | `cleverly` multi-arm selector C-TMLE | -0.0015 to 0.0025 | 0.9375 | 0.9591 | pass |
| three-arm binary-outcome law, discrete selector | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0071 to 0.0158 | 0.9413 | 0.9855 | pass |
| three-arm binary-outcome law, discrete selector | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0041 to 0.0196 | 0.9387 | 0.9780 | pass |
| three-arm binary-outcome law, discrete selector | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0037 to 0.0089 | 0.9437 | 0.9854 | pass |
| three-arm binary-outcome law, discrete selector | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0021 to 0.0095 | 0.9425 | 0.9845 | pass |
| three-arm binary-outcome law, greedy selector | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` multi-arm selector C-TMLE | -0.0044 to 0.0013 | 0.9363 | 0.9580 | pass |
| three-arm binary-outcome law, greedy selector | `ate[medium vs high]` | difference in counterfactual means, medium versus high | `cleverly` multi-arm selector C-TMLE | -0.0041 to 0.0013 | 0.9400 | 0.9877 | pass |
| three-arm binary-outcome law, greedy selector | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` multi-arm selector C-TMLE | -0.0012 to 0.0028 | 0.9513 | 0.9824 | pass |
| three-arm binary-outcome law, greedy selector | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` multi-arm selector C-TMLE | -0.0027 to 0.0013 | 0.9413 | 0.9898 | pass |
| three-arm binary-outcome law, greedy selector | `ey[medium]` | counterfactual mean under treatment arm 'medium' | `cleverly` multi-arm selector C-TMLE | -0.0025 to 0.0014 | 0.9437 | 0.9704 | pass |
| three-arm binary-outcome law, greedy selector | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0169 to 0.0066 | 0.9387 | 0.9578 | pass |
| three-arm binary-outcome law, greedy selector | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0153 to 0.0080 | 0.9450 | 0.9875 | pass |
| three-arm binary-outcome law, greedy selector | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0093 to 0.0037 | 0.9375 | 0.9567 | pass |
| three-arm binary-outcome law, greedy selector | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0077 to 0.0038 | 0.9450 | 0.9878 | pass |
| three-arm binary-outcome law, ordered selector | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` multi-arm selector C-TMLE | -0.0026 to 0.0029 | 0.9450 | 0.9906 | pass |
| three-arm binary-outcome law, ordered selector | `ate[medium vs high]` | difference in counterfactual means, medium versus high | `cleverly` multi-arm selector C-TMLE | -0.0028 to 0.0025 | 0.9463 | 1.0012 | pass |
| three-arm binary-outcome law, ordered selector | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` multi-arm selector C-TMLE | -0.0011 to 0.0029 | 0.9363 | 0.9664 | pass |
| three-arm binary-outcome law, ordered selector | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` multi-arm selector C-TMLE | -0.000906 to 0.0031 | 0.9500 | 0.9986 | pass |
| three-arm binary-outcome law, ordered selector | `ey[medium]` | counterfactual mean under treatment arm 'medium' | `cleverly` multi-arm selector C-TMLE | -0.0011 to 0.0027 | 0.9450 | 1.0006 | pass |
| three-arm binary-outcome law, ordered selector | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0098 to 0.0130 | 0.9437 | 0.9911 | pass |
| three-arm binary-outcome law, ordered selector | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0097 to 0.0134 | 0.9500 | 1.0028 | pass |
| three-arm binary-outcome law, ordered selector | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0058 to 0.0067 | 0.9363 | 0.9895 | pass |
| three-arm binary-outcome law, ordered selector | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0058 to 0.0057 | 0.9487 | 0.9914 | pass |
<!-- /generated -->

## Canonical comparison

No canonical implementation is compared for this row. The committed `equivalence.csv` is
therefore intentionally empty.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `fold_policy` | `treatment_stratified` | diagnostic | each split is balanced on the treatment, which the package refuses | reported, not gated: the coverage interval and its paired difference from the unstratified arm are published and no margin is applied | coverage 0.9331 to 0.9469, paired coverage difference -0.0069 to 0.0016 | reported |
| `fold_policy` | `unstratified` | diagnostic | each split is drawn without reading the treatment or the outcome | reported, not gated: the coverage interval is published and no margin is applied | coverage 0.9359 to 0.9494 | reported |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9119 to 0.9454, SE ratio 0.8987 to 0.9862 | **fail** |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000089, coverage 0.9087 to 0.9702, SE ratio 0.9567 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.0010, coverage 0.8965 to 0.9627, SE ratio 0.9892 | **fail** |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000647, coverage 0.9057 to 0.9683, SE ratio 0.9815 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5502 to -0.4599 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5123 to -0.5058 | pass |
| `selector_necessity` | `discrete` | positive | the discrete selector chooses among a declared candidate ladder | bias interval inside the equivalence margin, and RMSE below the control's by the declared ratio | bias 0.1589 to 0.1660, margin 0.0069, RMSE ratio 1 | **fail** |
| `selector_necessity` | `empty_control` | control | the selector is forced to stop at an empty path | bias interval must fall entirely outside the margin | bias 0.1589 to 0.1660, margin 0.0069, RMSE ratio 1 | pass |
| `selector_necessity` | `greedy` | positive | the greedy selector chooses its own mechanism path | bias interval inside the equivalence margin, and RMSE below the control's by the declared ratio | bias 0.0279 to 0.0443, margin 0.0158, RMSE ratio 0.4415 | **fail** |
| `selector_necessity` | `ordered` | positive | the ordered selector chooses how far along a fixed covariate order to go | bias interval inside the equivalence margin, and RMSE below the control's by the declared ratio | bias 0.0154 to 0.0304, margin 0.0145, RMSE ratio 0.3781 | **fail** |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0625, 0.0354 to 0.1004 | **fail** |
<!-- /generated -->

## Measured values

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications per selector strategy |
| `n` | 1500 | observations per primary replication |
| `independent_tests_total` | 27 | implementation-estimand truth tests |
| `independent_tests_passed` | 27 | truth tests passing |
| `subject_tests_total` | 27 | selector truth tests |
| `subject_tests_passed` | 27 | selector truth tests passing |
| `property_cells_total` | 12 | repeated-sampling property cells |
| `property_cells_passed` | 5 | property cells passing |
| `property_cells_reported` | 2 | fold-policy rows reported rather than gated, and excluded from the two counts above |
| `properties[fold_policy/unstratified]:coverage` | 0.9429 | reported coverage of the split this row declares |
| `properties[fold_policy/treatment_stratified]:coverage` | 0.9403 | reported coverage of the treatment-stratified split |
| `properties[fold_policy/treatment_stratified]:coverage_gain_ci_lower` | -0.0069 | its paired 99% lower bound against the unstratified arm |
| `properties[fold_policy/treatment_stratified]:coverage_gain_ci_upper` | 0.0016 | its paired 99% upper bound against the unstratified arm |
| `properties[fold_policy/treatment_stratified]:replicates` | 8000 | paired replications behind that difference |
| `properties[root_n_and_efficiency/n_500]:replicates` | 400 | replications behind the gated n = 500 cell |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal estimator size |
| `margin:nominal_coverage` | 0.9500 | nominal estimator coverage |
| `margin:bootstrap_replicates` | 10000 | resamples for bootstrap intervals |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin in empirical SDs |
| `margin:coverage_floor` | 0.9000 | lower coverage gate |
| `margin:over_coverage_ceiling` | 0.9900 | reported conservative-coverage ceiling |
| `margin:se_ratio_sanity_lower` | 0.8000 | lower SE-ratio screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | upper SE-ratio screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | lower calibration band |
| `margin:calibration_se_ratio_upper` | 1.0700 | upper calibration band |
| `margin:calibration_coverage_lower` | 0.9200 | lower calibration-coverage band |
| `margin:calibration_coverage_upper` | 0.9800 | upper calibration-coverage band |
| `margin:type_i_ceiling` | 0.1000 | largest supported type-I rate |
| `margin:paired_difference` | 0.1500 | paired similarity margin in empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio non-inferiority margin |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference non-inferiority margin |
| `margin:calibration_noninferiority` | 0.0500 | SE-calibration non-inferiority margin |
| `margin:minimum_power` | 0.8000 | lower power gate |
| `margin:root_n_slope` | -0.5000 | root-n contraction target |
| `margin:root_n_slope_lower` | -0.6250 | lower contraction band |
| `margin:root_n_slope_upper` | -0.3750 | upper contraction band |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:selector_rmse_ratio` | 0.8000 | required collaborative-to-control RMSE ratio |

## Limitations

This row has reporting policy because multi-arm repeated-sampling evidence was absent when the
study was declared. Six of its cells are red, and each one measures a different thing.

The selector-necessity law is a strong instrument. It is the only law in this row that puts units
outside the declared 0.025 truncation bounds.
`tests/unit/test_multi_arm_studies.py::test_the_selector_law_leaves_the_declared_truncation_region`
records that difference against the registered law, which keeps every unit inside them. The greedy
and ordered paths cut the error ratio well below the declared bar on that law. Both paths still
carry a bias the equivalence margin rejects at n = 1,500, and their coverage stays under the floor.
The discrete ladder reaches a ratio of one, because its cross-validated loss selects the empty
candidate on every draw. Read those cells as measurements under practical positivity loss. They do
not transfer to a law that satisfies the estimator's declared bounds.

The interval-calibration and null-size cells are red on the registered law, where the selector
reaches the empty candidate under a correct outcome regression. The reported covariance treats the
selected candidate as fixed and makes no conditional-coverage claim, so this row offers no
calibrated inference while selection is load-bearing.
[F18](../../roadmap.md#f18-selector-path-c-tmle-inference) records the missing result.

The n = 500 size cell is red as well. Its exact 99% coverage endpoint falls below the 0.90 floor
by a few thousandths at 400 replications. Every other declared quantity in that cell is green:
the bias sits well inside the margin and the SE ratio sits inside the sanity band. The cell was
green on a treatment-stratified split and is red on the unstratified one this row now declares.
The budget, the size, the seed and the margins did not move, so the comparison is between two
splits at one Monte Carlo resolution. The two larger sizes stay green, and both root-n rates
stay inside their band, so the row records no departure from the rate the theory predicts.

The `fold_policy` family now measures that comparison. It runs the two split policies on one
binary law and on one set of draws. It reports and it does not gate. Neither cell declares a
margin, so the pass counts above leave both out.

The measured table above gives the paired coverage difference and its two endpoints. That
interval covers zero. The fold policy therefore does not explain the endpoint move on this law
at this size.

The diagnostic reads its own coverage over 8,000 draws. That coverage does not re-read the
`n_500` cell. The gated cell keeps its own 400-replication budget.
[RM18](../../roadmap.md#rm18-red-property-cells-after-the-fold-scale-and-law-changes) declares
the rule this reading applies.

The row does not establish equivalence to an external package, simultaneous inference, conditional
effects, or cross-fitted primary performance. It covers binary outcomes, ordinary GLM nuisance
fits, and pointwise intervals. It excludes missing outcomes, weights, clusters, fold repeats,
simultaneous bands, and longitudinal treatment.
