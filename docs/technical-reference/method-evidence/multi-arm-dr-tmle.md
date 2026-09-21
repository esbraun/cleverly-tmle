# Multi-arm DR-TMLE

This study exercises the armwise multi-treatment extension implemented by `cleverly` and R
[`drtmle`](https://github.com/benkeser/drtmle) 1.1.2 at pinned commit `538a3a2`. The two
implementations receive identical samples, one exact five-fold assignment, and the same Python
out-of-fold initial outcome and treatment predictions. That isolates the reduced regressions,
corrections, targeting, and inference from nuisance-fit differences. `random_partition` draws that
assignment from the row count and the sample's seed, so it reads neither the arm labels nor the
outcome.

The binary-outcome law reports three arm means and ATE, risk-ratio, and odds-ratio contrasts
against the `high` reference arm. A separate fit-diagnostics artifact audits empirical scores
and solver status for every replication.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` multi-arm DR-TMLE | -0.0026 to 0.0020 | 0.9550 | 1.0632 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[low vs high]` | difference in counterfactual means, low versus high | R `drtmle` multi-arm extension | -0.0026 to 0.0020 | 0.9550 | 1.0636 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[medium vs high]` | difference in counterfactual means, medium versus high | `cleverly` multi-arm DR-TMLE | -0.0030 to 0.0019 | 0.9537 | 1.0037 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[medium vs high]` | difference in counterfactual means, medium versus high | R `drtmle` multi-arm extension | -0.0030 to 0.0018 | 0.9537 | 1.0038 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` multi-arm DR-TMLE | -0.000878 to 0.0027 | 0.9525 | 0.9990 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[high]` | counterfactual mean under treatment arm 'high' | R `drtmle` multi-arm extension | -0.000876 to 0.0027 | 0.9525 | 0.9989 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` multi-arm DR-TMLE | -0.0011 to 0.0023 | 0.9600 | 1.0413 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[low]` | counterfactual mean under treatment arm 'low' | R `drtmle` multi-arm extension | -0.0011 to 0.0023 | 0.9600 | 1.0421 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[medium]` | counterfactual mean under treatment arm 'medium' | `cleverly` multi-arm DR-TMLE | -0.0014 to 0.0020 | 0.9637 | 1.0366 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[medium]` | counterfactual mean under treatment arm 'medium' | R `drtmle` multi-arm extension | -0.0014 to 0.0020 | 0.9637 | 1.0367 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` multi-arm DR-TMLE | -0.0099 to 0.0090 | 0.9550 | 1.0611 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | R `drtmle` multi-arm extension | -0.0099 to 0.0090 | 0.9550 | 1.0615 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | `cleverly` multi-arm DR-TMLE | -0.0111 to 0.0099 | 0.9550 | 1.0032 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | R `drtmle` multi-arm extension | -0.0112 to 0.0099 | 0.9550 | 1.0032 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` multi-arm DR-TMLE | -0.0058 to 0.0048 | 0.9537 | 1.0509 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | R `drtmle` multi-arm extension | -0.0058 to 0.0048 | 0.9537 | 1.0512 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | `cleverly` multi-arm DR-TMLE | -0.0062 to 0.0042 | 0.9525 | 0.9959 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | R `drtmle` multi-arm extension | -0.0062 to 0.0042 | 0.9513 | 0.9959 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[low vs high]` | difference in counterfactual means, low versus high | 0.000004 | 0.0011 | 1.0005 | 0 | 0.000613 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[medium vs high]` | difference in counterfactual means, medium versus high | 0.000003 | 0.000783 | 1.0004 | 0 | 0.000439 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[high]` | counterfactual mean under treatment arm 'high' | -0.000002 | 0.000684 | 1.0001 | 0 | 0.000263 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[low]` | counterfactual mean under treatment arm 'low' | 0.000002 | 0.000761 | 1.0010 | 0 | 0.0018 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[medium]` | counterfactual mean under treatment arm 'medium' | 0.000001 | 0.000416 | 1.0007 | 0 | 0.000528 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | 0.000031 | 0.0012 | 1.0005 | 0 | 0.000619 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | 0.000046 | 0.000879 | 1.0005 | 0 | 0.000450 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | 0.000012 | 0.0010 | 1.0004 | 0 | 0.000641 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | 0.000012 | 0.000795 | 1.0004 | 0.0012 | 0.000255 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robust_contraction` | `both_wrong_n2000` | control | both nuisances are misspecified, at n = 2,000 | the exact coverage interval must fall below the floor, over the replications this rung declares its own verdict at | coverage 0.4683 to 0.5747, bias 0.0503 | pass |
| `double_robust_contraction` | `both_wrong_n4000` | control | both nuisances are misspecified, at n = 4,000 | the exact coverage interval must fall below the floor, over the replications this rung declares its own verdict at | coverage 0.2089 to 0.3018, bias 0.0487 | pass |
| `double_robust_contraction` | `both_wrong_n8000` | control | both nuisances are misspecified, at n = 8,000 | the exact coverage interval must fall below the floor, over the replications this rung declares its own verdict at | coverage 0.0235 to 0.0674, bias 0.0490 | pass |
| `double_robust_contraction` | `outcome_correct_n2000` | positive | only the outcome regression is correctly specified, at n = 2,000 | the exact coverage interval clears the declared floor, over the replications this rung declares its own verdict at | coverage 0.9245 to 0.9714, bias 0.0011 | pass |
| `double_robust_contraction` | `outcome_correct_n4000` | positive | only the outcome regression is correctly specified, at n = 4,000 | the exact coverage interval clears the declared floor, over the replications this rung declares its own verdict at | coverage 0.8988 to 0.9542, bias 0.000900 | **fail** |
| `double_robust_contraction` | `outcome_correct_n8000` | positive | only the outcome regression is correctly specified, at n = 8,000 | the exact coverage interval clears the declared floor, over the replications this rung declares its own verdict at | coverage 0.9165 to 0.9662, bias 0.0017 | pass |
| `double_robust_contraction` | `rate_both_wrong` | control | the same regression with both nuisances misspecified | slope interval must not establish contraction | slope -0.0619 to 0.0258 | pass |
| `double_robust_contraction` | `rate_outcome_correct` | positive | log absolute bias regressed on log n across three sizes, outcome regression correct, over every replication each rung ran | slope interval entirely below zero, so the bias contracts | slope -1.2342 to 3.7735 | **fail** |
| `double_robust_contraction` | `rate_treatment_correct` | positive | the same regression with only the treatment mechanism correct | slope interval entirely below zero, so the bias contracts | slope -3.2412 to 3.3387 | **fail** |
| `double_robust_contraction` | `treatment_correct_n2000` | positive | only the treatment mechanism is correctly specified, at n = 2,000 | the exact coverage interval clears the declared floor, over the replications this rung declares its own verdict at | coverage 0.9204 to 0.9688, bias 0.000585 | pass |
| `double_robust_contraction` | `treatment_correct_n4000` | positive | only the treatment mechanism is correctly specified, at n = 4,000 | the exact coverage interval clears the declared floor, over the replications this rung declares its own verdict at | coverage 0.9224 to 0.9701, bias -0.000437 | pass |
| `double_robust_contraction` | `treatment_correct_n8000` | positive | only the treatment mechanism is correctly specified, at n = 8,000 | the exact coverage interval clears the declared floor, over the replications this rung declares its own verdict at | coverage 0.9285 to 0.9740, bias 0.000830 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0052 to 0.0024, margin 0.0090, SE ratio 1.0625 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0466 to 0.0544, margin 0.0092, SE ratio 1.0265 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000511 to 0.0076, margin 0.0096, SE ratio 0.9980 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0015 to 0.0072, margin 0.0068, SE ratio 0.9998 | **fail** |
| `fold_policy` | `treatment_stratified` | diagnostic | each split is balanced on the treatment, which the package refuses | reported, not gated: the coverage interval and its paired difference from the unstratified arm are published and no margin is applied | coverage 0.9451 to 0.9576, paired coverage difference -0.0056 to 0.0045 | reported |
| `fold_policy` | `unstratified` | diagnostic | each split is drawn without reading the treatment or the outcome | reported, not gated: the coverage interval is published and no margin is applied | coverage 0.9458 to 0.9582 | reported |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9300 to 0.9597, SE ratio 0.9258 to 1.0123 | **fail** |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000290, coverage 0.9307 to 0.9826, SE ratio 1.0010 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.000902, coverage 0.8965 to 0.9627, SE ratio 0.9888 | **fail** |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000534, coverage 0.9212 to 0.9774, SE ratio 1.0454 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5835 to -0.4823 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5171 to -0.5090 | pass |
<!-- /generated -->

## Measured values

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 18 | implementation-estimand truth tests |
| `independent_tests_passed` | 18 | truth tests passing |
| `paired_tests_total` | 9 | paired comparison tests |
| `paired_tests_passed` | 9 | paired tests passing |
| `property_cells_total` | 22 | repeated-sampling property cells |
| `property_cells_passed` | 16 | property cells passing |
| `property_cells_reported` | 2 | fold-policy rows reported rather than gated, and excluded from the two counts above |
| `score_audited_fits` | 800 | fits each side audits against the shared score bar |
| `subject_score_failures` | 2 | Cleverly fits that miss that bar |
| `reference_score_failures` | 4 | R `drtmle` fits that miss that bar |
| `properties[fold_policy/unstratified]:coverage` | 0.9523 | reported coverage of the split this row declares |
| `properties[fold_policy/treatment_stratified]:coverage` | 0.9516 | reported coverage of the treatment-stratified split |
| `properties[fold_policy/treatment_stratified]:coverage_gain_ci_lower` | -0.0056 | its paired 99% lower bound against the unstratified arm |
| `properties[fold_policy/treatment_stratified]:coverage_gain_ci_upper` | 0.0045 | its paired 99% upper bound against the unstratified arm |
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
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |

## Limitations

This is a reporting study. A red scientific verdict publishes rather than prevents the record
from existing. The source theorem and the original R study are binary-treatment results. Both
packages expose a vector intervention API, and this row measures their armwise extension. It does
not claim a new multi-arm theorem.

The contraction ladder explains most of the red cells above. The `treatment_correct` level cell
misses its equivalence margin at n = 2,000 by 0.0004, which is about six per cent of the margin.
The two figures print four units of the last digit apart in the generated table above.
The ladder redraws that regime on independent streams at three sizes and puts the bias inside the
margin at every one, so the level cell records Monte Carlo error rather than a remainder.

Both fitted slopes then have nothing to regress. A one-correct bias that sits at the noise floor
and changes sign across the ladder gives a wide slope interval, and neither interval establishes
contraction. Read the ladder through its coverage rungs instead. Those rungs hold near the
nominal rate in both one-correct regimes, while the both-wrong control's coverage collapses as
the sample grows and its bias does not move.

The three remaining red cells are red on their own terms, and each misses by less than the width
of its own last printed digit. Six property rows are red in all: these three, the
`treatment_correct` level cell above, and the two contraction slopes. The `interval_calibration` SE-ratio interval reaches `0.9258` against a band that
opens at `0.93`. The `outcome_correct` rung at n = 4,000 has a coverage lower bound of `0.8988`
against a floor of `0.90`. The `n_500` size cell has a lower bound of `0.8965` against the same
floor.

The `n_500` cell was green at `0.9057` on the treatment-stratified split this row used before.
Nothing but the fold draw moved, and 400 replications at n = 500 do not resolve a coverage
endpoint to better than about a point. Read it as one Monte Carlo resolution rather than as a
property of the estimator. The same cell in the outcome-adaptive multi-arm row moved the other way
under the same change.

The `fold_policy` family now measures that comparison. It runs the two split policies on one
binary law and on one set of draws. It reports and it does not gate. Neither cell declares a
margin, so the pass counts above leave both out.

The measured table above gives the paired coverage difference and its two endpoints. That
interval covers zero. The upper endpoint stays below the threshold the declared rule names. The
fold policy does not explain the endpoint move on this law at this size.

This reading has less room than its design asked for. The declaration budgeted 8,000 draws for a
99% half-width of 0.0046, which would resolve the 0.005 the rule names. The realized half-width
is 0.0051, because the two policies disagreed on more draws than the pilot predicted. The upper
endpoint clears 0.005 only because the point estimate fell below zero. So this reading rests on
where the difference landed, and not on the resolution the instrument achieved.
[RM18](../../roadmap.md#rm18-red-property-cells-after-the-fold-scale-and-law-changes) records the
declared and realized numbers.

The diagnostic reads its own coverage over 8,000 draws. That coverage does not re-read the
`n_500` cell. The gated cell keeps its own 400-replication budget.
[RM18](../../roadmap.md#rm18-red-property-cells-after-the-fold-scale-and-law-changes) declares
the rule this reading applies.

Each side audits `score_audited_fits` fits against the shared empirical-score bar. Cleverly misses
it on `subject_score_failures` fits, and R `drtmle` misses it on `reference_score_failures`. The
propensity bound stays inactive throughout, and the subject fit is refused if it does not.

The row covers one binary-outcome law, one fold count, pooled reduced cross-fitting, univariate
reduction, ordinary GLM nuisance fits, and pointwise intervals. It excludes flexible learners,
practical-positivity stress, missing outcomes, weights, clusters, fold repeats, simultaneous
bands, and longitudinal treatment.
