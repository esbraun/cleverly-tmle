# Randomized multi-arm missing-outcome DR-TMLE

This study validates missing-outcome DR-TMLE at three arms. The law is L3 of
`tests/studies/mar_arm_indexed_laws.py`: three arms named `low`, `mid` and `high`, a binary
outcome, and a response mechanism that depends on treatment and a baseline covariate. The fit
receives the W-stratified treatment table as known probabilities, keyed by level. The study
compares the both-correct limit with R `drtmle` 1.1.2 at pinned commit
[`538a3a2`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c).

The comparator boundary matters. R `drtmle` fits one joint treatment-response mechanism with the
composite response `A == a & DeltaA == 1 & DeltaY == 1`. Díaz and van der Laan (2017, page 25)
reject that composite construction for this problem. `cleverly` keeps the treatment and
observation mechanisms apart, with five reductions for each arm and a separate tilt for each arm.
So the paired comparison shows the shared both-correct limit only. The property cells test the
armwise construction itself.

This study is the only measurement of the default simultaneous band at more than two arms for
this estimator. The `default-simultaneous-bands` study measures the two-arm band alone.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` randomized multi-arm missing-outcome DR-TMLE | -0.0017 to 0.0053 | 0.9413 | 0.9846 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ate[low vs high]` | difference in counterfactual means, low versus high | R `drtmle` at three arms with a joint treatment-response mechanism | -0.0017 to 0.0053 | 0.9400 | 0.9851 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ate[mid vs high]` | difference in counterfactual means, mid versus high | `cleverly` randomized multi-arm missing-outcome DR-TMLE | -0.0023 to 0.0043 | 0.9587 | 1.0052 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ate[mid vs high]` | difference in counterfactual means, mid versus high | R `drtmle` at three arms with a joint treatment-response mechanism | -0.0023 to 0.0043 | 0.9575 | 1.0052 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` randomized multi-arm missing-outcome DR-TMLE | -0.0035 to 0.0014 | 0.9400 | 0.9827 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ey[high]` | counterfactual mean under treatment arm 'high' | R `drtmle` at three arms with a joint treatment-response mechanism | -0.0036 to 0.0014 | 0.9413 | 0.9829 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` randomized multi-arm missing-outcome DR-TMLE | -0.0019 to 0.0033 | 0.9300 | 0.9557 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ey[low]` | counterfactual mean under treatment arm 'low' | R `drtmle` at three arms with a joint treatment-response mechanism | -0.0019 to 0.0033 | 0.9300 | 0.9561 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ey[mid]` | counterfactual mean under treatment arm 'mid' | `cleverly` randomized multi-arm missing-outcome DR-TMLE | -0.0022 to 0.0021 | 0.9563 | 1.0170 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ey[mid]` | counterfactual mean under treatment arm 'mid' | R `drtmle` at three arms with a joint treatment-response mechanism | -0.0023 to 0.0020 | 0.9563 | 1.0172 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` randomized multi-arm missing-outcome DR-TMLE | -0.0123 to 0.0189 | 0.9450 | 0.9832 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | R `drtmle` at three arms with a joint treatment-response mechanism | -0.0122 to 0.0190 | 0.9437 | 0.9837 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `or[mid vs high]` | marginal odds ratio, mid versus high, reported on the log scale | `cleverly` randomized multi-arm missing-outcome DR-TMLE | -0.0124 to 0.0165 | 0.9600 | 1.0009 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `or[mid vs high]` | marginal odds ratio, mid versus high, reported on the log scale | R `drtmle` at three arms with a joint treatment-response mechanism | -0.0123 to 0.0165 | 0.9587 | 1.0010 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` randomized multi-arm missing-outcome DR-TMLE | -0.0062 to 0.0090 | 0.9363 | 0.9750 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | R `drtmle` at three arms with a joint treatment-response mechanism | -0.0061 to 0.0090 | 0.9363 | 0.9756 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `rr[mid vs high]` | marginal risk ratio, mid versus high, reported on the log scale | `cleverly` randomized multi-arm missing-outcome DR-TMLE | -0.0042 to 0.0066 | 0.9613 | 1.0102 | pass |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `rr[mid vs high]` | marginal risk ratio, mid versus high, reported on the log scale | R `drtmle` at three arms with a joint treatment-response mechanism | -0.0042 to 0.0066 | 0.9613 | 1.0102 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ate[low vs high]` | difference in counterfactual means, low versus high | -0.000019 | 0.0033 | 1.0021 | 0.0013 | 0.0016 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ate[mid vs high]` | difference in counterfactual means, mid versus high | 9.748e-07 | 0.000181 | 1.0011 | 0.0012 | 0.0015 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ey[high]` | counterfactual mean under treatment arm 'high' | 0.000027 | 0.0067 | 1.0011 | -0.0013 | 0.0014 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ey[low]` | counterfactual mean under treatment arm 'low' | 0.000008 | 0.0019 | 1.0028 | 0 | 0.0021 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `ey[mid]` | counterfactual mean under treatment arm 'mid' | 0.000028 | 0.0079 | 1.0014 | 0 | 0.0017 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | -0.000023 | 0.0032 | 1.0024 | 0.0012 | 0.0016 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `or[mid vs high]` | marginal odds ratio, mid versus high, reported on the log scale | -0.000009 | 0.000705 | 1.0016 | 0.0012 | 0.0015 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | -0.000010 | 0.0015 | 1.0026 | 0 | 0.0018 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment | `rr[mid vs high]` | marginal risk ratio, mid versus high, reported on the log scale | 0.000009 | 0.0012 | 1.0012 | 0 | 0.0015 vs 0.0500 | equivalent |
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `corrected_mar_inference` | `l3_ate_low__both_correct` | positive | L3 difference, low versus high: the outcome regression and observation mechanism are correctly specified | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0026 to 0.0043, coverage 0.9282 to 0.9688, SE ratio 1.0110 | pass |
| `corrected_mar_inference` | `l3_ate_low__both_wrong` | control | L3 difference, low versus high: the outcome regression and observation mechanism are both misspecified | bias interval must fall entirely outside the margin, and its distance from zero must clear the declared floor, SE ratio must remain between 0.1 and 10.0 | bias 0.0129 to 0.0201, coverage 0.8878 to 0.9396, SE ratio 0.9697 | pass |
| `corrected_mar_inference` | `l3_ate_low__observation_drift` | positive | L3 difference, low versus high: the outcome regression is correct and the observation mechanism is misspecified | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0030 to 0.0040, coverage 0.9238 to 0.9657, SE ratio 0.9644 | pass |
| `corrected_mar_inference` | `l3_ate_low__outcome_drift` | positive | L3 difference, low versus high: the outcome regression is misspecified and the observation mechanism is correct | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0043 to 0.0027, coverage 0.9150 to 0.9596, SE ratio 0.9815 | pass |
| `corrected_mar_inference` | `l3_ate_mid__both_correct` | positive | L3 difference, mid versus high: the outcome regression and observation mechanism are correctly specified | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0038 to 0.0026, coverage 0.9371 to 0.9747, SE ratio 1.0262 | pass |
| `corrected_mar_inference` | `l3_ate_mid__both_wrong` | control | L3 difference, mid versus high: the outcome regression and observation mechanism are both misspecified | bias interval must fall entirely outside the margin, and its distance from zero must clear the declared floor, SE ratio must remain between 0.1 and 10.0 | bias -0.0242 to -0.0168, coverage 0.8585 to 0.9167, SE ratio 0.9312 | pass |
| `corrected_mar_inference` | `l3_ate_mid__observation_drift` | positive | L3 difference, mid versus high: the outcome regression is correct and the observation mechanism is misspecified | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0028 to 0.0038, coverage 0.9208 to 0.9637, SE ratio 0.9957 | pass |
| `corrected_mar_inference` | `l3_ate_mid__outcome_drift` | positive | L3 difference, mid versus high: the outcome regression is misspecified and the observation mechanism is correct | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0044 to 0.0020, coverage 0.9356 to 0.9737, SE ratio 1.0466 | pass |
| `correction_necessity` | `five_reduction_cycle__closed_score` | positive | five-reduction correction cycle: the correction scores after the complete five-reduction cycle | the upper confidence endpoint is below the declared fraction of the initial-score lower endpoint | score 7.012e-10 to 7.545e-08 | pass |
| `correction_necessity` | `five_reduction_cycle__initial_score_control` | control | five-reduction correction cycle: the same correction scores before the cycle is run | the lower confidence endpoint clears the declared unresolved-score floor | score 0.0104 to 0.0114 | pass |
| `interval_calibration` | `ate__correctly_specified` | positive | average treatment effect: all three required nuisance functions are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9287 to 0.9537, SE ratio 0.9294 to 1.0000, empirical efficiency ratio 0.9994 to 1.0751, reported efficiency ratio 0.9980 to 1.0008 | **fail** |
| `interval_calibration` | `ate__noise_control` | control | average treatment effect: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8081 to 0.8481, SE ratio 0.6664 to 0.7185, empirical efficiency ratio 1.3910 to 1.5002, reported efficiency ratio 0.9979 to 1.0008 | pass |
| `interval_calibration` | `ate__shrunken_se_control` | control | average treatment effect: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8003 to 0.8409, SE ratio 0.6523 to 0.7004, empirical efficiency ratio 0.9989 to 1.0725, reported efficiency ratio 0.6986 to 0.7006 | pass |
| `power` | `l3_ate_mid__randomized_alternative` | positive | L3 difference, mid versus high: the same test applied to the randomized law's own nonzero contrast | rejection lower bound clears the minimum power | rejection 0.9800, 0.9671 to 0.9889 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.0013, coverage 0.9353 to 0.9676, SE ratio 1.0027 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.000473, coverage 0.9392 to 0.9704, SE ratio 1.0187 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000785, coverage 0.9344 to 0.9669, SE ratio 1.0029 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5201 to -0.4675 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5008 to -0.4977 | pass |
| `simultaneous_coverage` | `arms__pointwise_joint_control` | control | the multi-arm fit's three arm means, two differences, two risk ratios and two odds ratios: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.7713 to 0.8142 | pass |
| `simultaneous_coverage` | `arms__simultaneous_band` | positive | the multi-arm fit's three arm means, two differences, two risk ratios and two odds ratios: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9196 to 0.9462 | **fail** |
| `type_i_error` | `l3_ate_mid__randomized_sharp_null` | positive | L3 difference, mid versus high: a randomized law whose two compared arms share one outcome regression, so the contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling, and coverage clears the floor | rejection 0.0583, 0.0423 to 0.0780 | pass |
<!-- /generated -->

The property grid reports both contrasts against `high` under four nuisance configurations. The
drift cells use a constant outcome regression and a constant observation mechanism, as the
two-arm study does. The grid also tests root-n contraction, interval calibration with its two
controls, the size and power of the Wald test, the five-reduction score reduction, and the joint
coverage of the band over all nine estimates.

Two positive cells are red, and both read the same 2,400 calibration fits. The SE-ratio interval
of `ate__correctly_specified` ends at 0.9294 against a floor of 0.93. The joint-coverage interval
of `arms__simultaneous_band` ends at 0.9196 against 0.92. The study publishes them red under the
`reporting` policy.

The owner `F4-calibration-draws` in the [roadmap](../../roadmap.md#red-cell-owners) records the
diagnosis. The spread of that draw set is
1.037 times its mean standard error. Two independent cells of the same configuration read 0.997
and 0.989. The oracle band covers as the package band does.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 18 | truth tests passing |
| `independent_tests_total` | 18 | truth tests reported |
| `paired_tests_passed` | 9 | paired comparisons passing |
| `paired_tests_total` | 9 | paired comparisons reported |
| `property_cells_passed` | 20 | property cells passing |
| `property_cells_total` | 22 | property cells reported |
| `max_standardized_bias` | 0.0467 | largest primary standardized bias |
| `min_coverage` | 0.9300 | lowest primary coverage |
| `max_margin_utilization` | 0.0079 | largest paired similarity-margin share |
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
| `margin:correction_score_ratio` | 0.0100 | maximum closed-to-initial score-endpoint ratio |
| `margin:uncorrected_score_floor` | 0.0010 | unresolved initial-score floor |

## Limits

- The study covers one law with three arms, a binary outcome, one three-level baseline covariate,
  a known W-stratified randomization, and an MAR response mechanism.
- Every study fit passes the treatment probabilities as known. A fit with `randomized=True`
  estimates the categorical mechanism instead. Only the unit tests in
  `tests/unit/test_drtmle_missing_multi_arm.py` cover that route.
- The initial outcome and observation regressions are finite-support oracles. The reduced
  regressions use linear and logistic models without cross-fitting.
- The R comparator establishes the shared both-correct limit only, not parity of the armwise
  five-reduction construction.
- The study excludes observational assignment, weights, clusters, missing treatment, MNAR
  outcomes, cross-fitting, and other DR-TMLE compositions.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_mar_multi_arm/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_mar_multi_arm/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_mar_multi_arm/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_mar_multi_arm/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_mar_multi_arm/properties.csv)
carry the protocol, provenance, and every published row.
