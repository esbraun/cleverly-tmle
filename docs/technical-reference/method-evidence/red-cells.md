# The red-cell ledger

This page lists every red verdict that a registered study publishes. It names the roadmap ask that
owns each one. `python -m tests.studies.evidence.red_cells` writes each table from the committed
results. `tests/unit/test_red_cell_ledger.py` checks each table against those results.

## What counts as red

A study publishes three kinds of verdict. The ledger reads each kind the way the regeneration
gates it.

| kind | source file | red when |
| --- | --- | --- |
| truth | `performance-tests.csv` | the row fails its truth gate. The ledger skips a comparator row when the study declares an accepted reference failure |
| paired | `equivalence.csv` | the paired conclusion is not `equivalent` or `superior`. The ledger computes the conclusion again from the committed endpoints of each leg, and it refuses a row whose committed verdict disagrees |
| property | `properties.csv` | the cell fails its own rule or the joint clause of its family. A `diagnostic` row states no verdict, so it is never red |

The ledger reads each verdict column as a boolean. It refuses a table whose verdict column holds
a blank or any other value, because a blank would otherwise read as a pass. The paired
`reference_valid` column repeats the comparator's truth row, so the truth kind covers it.

The ledger reads verdict rows only. Five study modules also define a `scientific_failures` hook.
The regeneration adds each hook's rows to the failures it gates or reports. These hooks audit
fits and publish no verdict cell, so the ledger does not read them.

| study | policy | what the hook reports | what holds its result |
| --- | --- | --- | --- |
| [DR-TMLE for binary complete data](canonical-dr-tmle.md) | `reporting` | the score audit of both implementations, and the solver flag of `cleverly` | the study page and `fit-diagnostics.csv` |
| [multi-arm point-treatment DR-TMLE](multi-arm-dr-tmle.md) | `reporting` | the same two audits | the study page and `fit-diagnostics.csv` |
| [ordinary missing-outcome natural-course TMLE](ordinary-missing-outcome-natural-course-tmle.md) | `gated` | the exact-equality probe of the scale workaround | `scale-probe.csv`. The test requires every row to pass |
| [missing-outcome attributable-effect TMLE](missing-outcome-attributable-effects-tmle.md) | `gated` | the exact-equality probe of the scale workaround on both R paths | `scale-probe.csv`. The test requires every row to pass |
| [stacked arm-indexed missing-outcome CV-TMLE](stacked-arm-indexed-missing-outcome-cvtmle.md) | `gated` | the same probe | `scale-probe.csv`. The test requires every row to pass |

`tests/unit/test_red_cell_ledger.py` fails when a study module defines a hook that this table
does not list.

## Who owns each red row

Each owner is an `id` in the [Red-cell owners](../../roadmap.md#red-cell-owners) table of the
roadmap.
[F18](../../roadmap.md#f18-selector-path-c-tmle-inference) and
[F19](../../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) wait on a
published result, or on a natural extension that meets the
[Eligibility](../../roadmap.md#eligibility) conditions.
[F27](../../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) owns the two
truth rows of the [learned-rule boundary study](learned-rule-cvtmle-boundary.md). Its table states
that the `exceptional` row has no published result and that the `weak_blip` row has one. Each
`RM18-` owner holds cells that a declared diagnostic read once. Its acceptance cell gives the
reading and the declaration that a future regeneration needs.

A red row stays red under the `reporting` policy. It is read again only under the condition its
owner names. The ledger moves no margin and changes no verdict.

Every `F18` row measures a working-mechanism plug-in diagnostic. The package does not publish that
diagnostic as inference: the greedy, ordered, and discrete C-TMLE paths refuse `ci`, `pvalue`, and
`std_error`. The cells keep their `std_error`, `ci_lower`, `ci_upper`, and `covered` columns, and
those columns now read `plugin_std_error` and `plugin_interval`. The two accessors call the same
body the refused ones call, so the reframing changed no number and no study was regenerated. Read a
red `F18` coverage cell as a measurement of a diagnostic, and not as a refused coverage claim.

The two red `F19` generated-design cells likewise measure `plugin_std_error` and
`plugin_interval` for the shipped joint outcome-adaptive fit. Their coverage and SE-ratio numbers
remain evidence about that diagnostic, not an inference claim. F19 needs a result for the exact
shared-multinomial, jointly targeted construction before those cells can support one.

The test fails on four states.

| state | why the test refuses it |
| --- | --- |
| a red row with no owner | nobody is on record to close it |
| an owner entry with no red row | the entry describes a verdict that is no longer red |
| an owner that the roadmap `id` column does not list | the owner points at no ask |
| a red row in a study registered as `gated` | a `gated` study refuses to publish a red verdict |

<!-- generated: red-cell-summary -->
| owner | red rows | studies |
| --- | --- | --- |
| [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) | 9 | selector-based point-treatment C-TMLE, selector-based multi-arm C-TMLE |
| [`F19`](../../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) | 2 | outcome-adaptive multi-arm C-TMLE |
| [`F27`](../../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) | 2 | CV-TMLE of the fold-local learned-rule value at exceptional and weak-blip laws |
| [`RM18-fixed-weights`](../../roadmap.md#red-cell-owners) | 2 | cross-fitted weighted end-of-study longitudinal TMLE |
| [`RM18-boundary`](../../roadmap.md#red-cell-owners) | 8 | selector-based multi-arm C-TMLE, DR-TMLE for binary complete data, multi-arm point-treatment DR-TMLE, cross-fitted weighted end-of-study longitudinal TMLE |
| [`RM18-one-sided-bias`](../../roadmap.md#red-cell-owners) | 3 | DR-TMLE for binary complete data, multi-arm point-treatment DR-TMLE |
| [`RM18-ordinary-weighted`](../../roadmap.md#red-cell-owners) | 6 | ordinary weighted end-of-study longitudinal TMLE |
| [`RM18-comparator-density`](../../roadmap.md#red-cell-owners) | 1 | ordinary continuous modified treatment policies |
| [`strata-boundary-mean`](../../roadmap.md#red-cell-owners) | 5 | ordinary point-treatment TMLE with baseline strata |
| [`band-finite-sample`](../../roadmap.md#red-cell-owners) | 5 | clustered cross-fitted end-of-study longitudinal TMLE, ordinary survival-curve longitudinal TMLE, default simultaneous bands across shipped fit shapes |
| [`F4-calibration-draws`](../../roadmap.md#red-cell-owners) | 2 | randomized multi-arm missing-outcome DR-TMLE |
| [`composite-high-arm`](../../roadmap.md#red-cell-owners) | 2 | observational missing-data DR-TMLE with a composite indicator |
| [`X8-logit-small-stratum`](../../roadmap.md#red-cell-owners) | 2 | stratified incremental and MSM targeting |
| [`X8-identity-small-stratum`](../../roadmap.md#red-cell-owners) | 2 | identity-link MSM with baseline strata |
| [`X8-drtmle-one-sided-bias`](../../roadmap.md#red-cell-owners) | 4 | DR-TMLE with baseline strata |
| [`X8-drtmle-small-stratum`](../../roadmap.md#red-cell-owners) | 3 | DR-TMLE with baseline strata |
| [`F1-power-design`](../../roadmap.md#red-cell-owners) | 1 | ordinary known stochastic categorical longitudinal TMLE |
| [`mtp-longitudinal-limits`](../../roadmap.md#red-cell-owners) | 3 | longitudinal modified treatment policies |
| [`mtp-point-calibration`](../../roadmap.md#red-cell-owners) | 1 | ordinary point modified treatment policies |
| [`F28`](../../roadmap.md#red-cell-owners) | 9 | clustered point-treatment CV-TMLE at unequal cluster sizes, clustered TMLE intervals on a t reference at few clusters |
| [`X20-bootstrap`](../../roadmap.md#red-cell-owners) | 3 | full-refit bootstrap and derived contrasts |
| total | 75 | 24 studies |
<!-- /generated -->

## The red rows

The "measured" column gives the endpoints that the failed verdict reads. A property row uses the
same text as the "measured" column of its study page.

<!-- generated: red-cells -->
| study | kind | row | role | fails by | measured | owner |
| --- | --- | --- | --- | --- | --- | --- |
| [ordinary point-treatment TMLE with baseline strata](stratified-point-treatment-tmle.md) | truth | `cleverly-stratified-tmle/stratified_binary/ey[1][V=2]` | cleverly-stratified-tmle | its truth gate | bias -0.0027 to 0.000813, coverage 0.9170, SE ratio 0.9850 | [`strata-boundary-mean`](../../roadmap.md#red-cell-owners) |
| [ordinary point-treatment TMLE with baseline strata](stratified-point-treatment-tmle.md) | truth | `tmle3-stratified/stratified_binary/ey[1][V=2]` | tmle3-stratified | its truth gate | bias -0.0027 to 0.000813, coverage 0.9170, SE ratio 0.9849 | [`strata-boundary-mean`](../../roadmap.md#red-cell-owners) |
| [ordinary point-treatment TMLE with baseline strata](stratified-point-treatment-tmle.md) | property | `interval_calibration/v2_ey1__correctly_specified` | positive | its own rule | coverage 0.9056 to 0.9344, SE ratio 0.9315 to 1.0094, empirical efficiency ratio 0.9753 to 1.0552, reported efficiency ratio 0.9742 to 0.9931 | [`strata-boundary-mean`](../../roadmap.md#red-cell-owners) |
| [ordinary point-treatment TMLE with baseline strata](stratified-point-treatment-tmle.md) | property | `simultaneous_coverage/crossfit_strata__simultaneous_band` | positive | its own rule | joint coverage 0.9038 to 0.9329 | [`strata-boundary-mean`](../../roadmap.md#red-cell-owners) |
| [ordinary point-treatment TMLE with baseline strata](stratified-point-treatment-tmle.md) | property | `simultaneous_coverage/strata__simultaneous_band` | positive | its own rule | joint coverage 0.8881 to 0.9194 | [`strata-boundary-mean`](../../roadmap.md#red-cell-owners) |
| [stratified incremental and MSM targeting](stratified-incremental-msm.md) | property | `interval_calibration/logit_v1_a__correctly_specified` | positive | its own rule | coverage 0.9283 to 0.9555, SE ratio 0.9273 to 1.0105 | [`X8-logit-small-stratum`](../../roadmap.md#red-cell-owners) |
| [stratified incremental and MSM targeting](stratified-incremental-msm.md) | property | `interval_calibration/logit_v2_a__correctly_specified` | positive | its own rule | coverage 0.9289 to 0.9560, SE ratio 0.7818 to 0.9387 | [`X8-logit-small-stratum`](../../roadmap.md#red-cell-owners) |
| [identity-link MSM with baseline strata](stratified-msm-identity.md) | paired | `stratified_msm_identity/msm[W][V=2]` | paired | inconclusive: calibration leg | difference -0.000049 to 0.000037 within 0.000694, RMSE ratio bound 1.0047 vs 1.1000, coverage difference bound -0.0220 vs -0.0250, calibration excess bound 0.0523 vs 0.0500, resolution 0.0476 | [`X8-identity-small-stratum`](../../roadmap.md#red-cell-owners) |
| [identity-link MSM with baseline strata](stratified-msm-identity.md) | property | `interval_calibration/identity_v2_a__correctly_specified` | positive | its own rule | coverage 0.9233 to 0.9515, SE ratio 0.9264 to 1.0042, empirical efficiency ratio 0.9925 to 1.0766, reported efficiency ratio 0.9937 to 1.0005 | [`X8-identity-small-stratum`](../../roadmap.md#red-cell-owners) |
| [DR-TMLE with baseline strata](stratified-dr-tmle.md) | truth | `cleverly-stratified-drtmle/stratified_both_correct/ey[1][V=2]` | cleverly-stratified-drtmle | its truth gate | bias 0.0023 to 0.0112, coverage 0.9000, SE ratio 0.9388 | [`X8-drtmle-small-stratum`](../../roadmap.md#red-cell-owners) |
| [DR-TMLE with baseline strata](stratified-dr-tmle.md) | truth | `drtmle-r-stratified/stratified_both_correct/ey[1][V=2]` | drtmle-r-stratified | its truth gate | bias 0.0022 to 0.0111, coverage 0.8925, SE ratio 0.9205 | [`X8-drtmle-small-stratum`](../../roadmap.md#red-cell-owners) |
| [DR-TMLE with baseline strata](stratified-dr-tmle.md) | property | `double_robustness/marginal_ate__treatment_correct` | positive | its own rule | bias 0.0136 to 0.0185, margin 0.0067, SE ratio 0.9902 | [`X8-drtmle-one-sided-bias`](../../roadmap.md#red-cell-owners) |
| [DR-TMLE with baseline strata](stratified-dr-tmle.md) | property | `double_robustness/v0_ate__treatment_correct` | positive | its own rule | bias 0.0094 to 0.0162, margin 0.0093, SE ratio 0.9505 | [`X8-drtmle-one-sided-bias`](../../roadmap.md#red-cell-owners) |
| [DR-TMLE with baseline strata](stratified-dr-tmle.md) | property | `double_robustness/v1_ate__treatment_correct` | positive | its own rule | bias 0.0123 to 0.0214, margin 0.0124, SE ratio 0.9364 | [`X8-drtmle-one-sided-bias`](../../roadmap.md#red-cell-owners) |
| [DR-TMLE with baseline strata](stratified-dr-tmle.md) | property | `double_robustness/v2_ate__treatment_correct` | positive | its own rule | bias 0.0179 to 0.0286, margin 0.0147, SE ratio 1.0627 | [`X8-drtmle-one-sided-bias`](../../roadmap.md#red-cell-owners) |
| [DR-TMLE with baseline strata](stratified-dr-tmle.md) | property | `interval_calibration/v2_ate__correctly_specified` | positive | its own rule | coverage 0.9069 to 0.9380, SE ratio 0.9294 to 1.0112, empirical efficiency ratio 1.0348 to 1.1237, reported efficiency ratio 1.0296 to 1.0653 | [`X8-drtmle-small-stratum`](../../roadmap.md#red-cell-owners) |
| [clustered point-treatment CV-TMLE at unequal cluster sizes](clustered-unequal-cv-tmle.md) | property | `clustered_inference/cluster_robust_fold_evaluated_cluster_covariate` | positive | its own rule | coverage 0.9182 to 0.9450, SE ratio 0.9453 to 1.0187, paired coverage gain 0.4021 to 0.4542 | [`F28`](../../roadmap.md#red-cell-owners) |
| [clustered point-treatment CV-TMLE at unequal cluster sizes](clustered-unequal-cv-tmle.md) | property | `clustered_inference/iid_control_fold_evaluated_cluster_covariate` | control | the family's joint clause | coverage 0.4781 to 0.5310, SE ratio 0.3326 to 0.3580, paired coverage gain 0.4021 to 0.4542 | [`F28`](../../roadmap.md#red-cell-owners) |
| [clustered TMLE intervals on a t reference at few clusters](clustered-few-cluster-tmle.md) | property | `few_cluster_reference/drtmle_crossfit__equal10__j10__iid_t_control` | control | its own rule | bias -0.0013 to 0.0105, coverage 0.8874 to 0.9121, SE ratio 0.8499 to 1.0008 | [`F28`](../../roadmap.md#red-cell-owners) |
| [clustered TMLE intervals on a t reference at few clusters](clustered-few-cluster-tmle.md) | property | `few_cluster_reference/drtmle_crossfit__equal10__j20__iid_t_control` | control | its own rule | bias -0.000551 to 0.0077, coverage 0.8646 to 0.8915, SE ratio 0.8119 to 0.9042 | [`F28`](../../roadmap.md#red-cell-owners) |
| [clustered TMLE intervals on a t reference at few clusters](clustered-few-cluster-tmle.md) | property | `few_cluster_reference/drtmle_crossfit__equal10__j30__iid_t_control` | control | its own rule | bias -0.0011 to 0.0057, coverage 0.8531 to 0.8810, SE ratio 0.7967 to 0.8764 | [`F28`](../../roadmap.md#red-cell-owners) |
| [clustered TMLE intervals on a t reference at few clusters](clustered-few-cluster-tmle.md) | property | `few_cluster_reference/drtmle_crossfit__unequal_informative__j10__iid_t_control` | control | its own rule | bias 0.0242 to 0.0394, coverage 0.8049 to 0.8363, SE ratio 0.6849 to 0.8247 | [`F28`](../../roadmap.md#red-cell-owners) |
| [clustered TMLE intervals on a t reference at few clusters](clustered-few-cluster-tmle.md) | property | `few_cluster_reference/tmle_cv_evaluation__unequal_informative__j10__t_reference` | positive | its own rule | bias 0.1146 to 0.1310, coverage 0.8940 to 0.9180, SE ratio 0.8297 to 0.8849 | [`F28`](../../roadmap.md#red-cell-owners) |
| [clustered TMLE intervals on a t reference at few clusters](clustered-few-cluster-tmle.md) | property | `few_cluster_reference/tmle_cv_evaluation__unequal_informative__j20__t_reference` | positive | its own rule | bias 0.0528 to 0.0638, coverage 0.9078 to 0.9302, SE ratio 0.8994 to 0.9575 | [`F28`](../../roadmap.md#red-cell-owners) |
| [clustered TMLE intervals on a t reference at few clusters](clustered-few-cluster-tmle.md) | property | `few_cluster_reference/tmle_cv_evaluation__unequal_informative__j30__t_reference` | positive | its own rule | bias 0.0300 to 0.0386, coverage 0.9224 to 0.9430, SE ratio 0.9391 to 0.9971 | [`F28`](../../roadmap.md#red-cell-owners) |
| [clustered cross-fitted end-of-study longitudinal TMLE](clustered-cross-fitted-longitudinal-tmle.md) | property | `simultaneous_coverage/clustered_regimens__simultaneous_band` | positive | its own rule | joint coverage 0.9191 to 0.9458 | [`band-finite-sample`](../../roadmap.md#red-cell-owners) |
| [selector-based point-treatment C-TMLE](selector-based-point-treatment-c-tmle.md) | property | `selector_necessity/collaborative` | positive | its own rule | bias 0.0015 to 0.0037, margin 0.0030, RMSE ratio 0.2410 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based point-treatment C-TMLE](selector-based-point-treatment-c-tmle.md) | property | `selector_necessity/empty_control` | control | the family's joint clause | bias 0.0495 to 0.0511, margin 0.0022, RMSE ratio 0.2410 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based point-treatment C-TMLE](selector-based-point-treatment-c-tmle.md) | property | `type_i_error/sharp_null` | positive | its own rule | rejection 0.0700, 0.0412 to 0.1095 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `interval_calibration/correctly_specified` | positive | its own rule | coverage 0.9119 to 0.9454, SE ratio 0.8987 to 0.9862 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `root_n_and_efficiency/n_500` | positive | its own rule | bias 0.0010, coverage 0.8965 to 0.9627, SE ratio 0.9892 | [`RM18-boundary`](../../roadmap.md#red-cell-owners) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `selector_necessity/discrete` | positive | its own rule | bias 0.1589 to 0.1660, margin 0.0069, RMSE ratio 1 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `selector_necessity/empty_control` | control | the family's joint clause | bias 0.1589 to 0.1660, margin 0.0069, RMSE ratio 1 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `selector_necessity/greedy` | positive | its own rule | bias 0.0279 to 0.0443, margin 0.0158, RMSE ratio 0.4415 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `selector_necessity/ordered` | positive | its own rule | bias 0.0154 to 0.0304, margin 0.0145, RMSE ratio 0.3781 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `type_i_error/sharp_null` | positive | its own rule | rejection 0.0625, 0.0354 to 0.1004 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [outcome-adaptive multi-arm C-TMLE](outcome-adaptive-multi-arm-c-tmle.md) | property | `generated_design/estimated` | positive | its own rule | coverage 0.9267 to 0.9677, SE ratio 0.9694 to 1.1026 | [`F19`](../../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| [outcome-adaptive multi-arm C-TMLE](outcome-adaptive-multi-arm-c-tmle.md) | property | `generated_design/oracle_design` | positive | its own rule | coverage 0.9282 to 0.9688, SE ratio 0.9797 to 1.1151 | [`F19`](../../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| [DR-TMLE for binary complete data](canonical-dr-tmle.md) | property | `double_robust_contraction/treatment_correct_n1500` | positive | its own rule | coverage 0.8864 to 0.9385, bias 0.0115 | [`RM18-boundary`](../../roadmap.md#red-cell-owners) |
| [DR-TMLE for binary complete data](canonical-dr-tmle.md) | property | `double_robustness/outcome_correct` | positive | its own rule | bias 0.0026 to 0.0075, margin 0.0067, SE ratio 0.9888 | [`RM18-one-sided-bias`](../../roadmap.md#red-cell-owners) |
| [DR-TMLE for binary complete data](canonical-dr-tmle.md) | property | `double_robustness/treatment_correct` | positive | its own rule | bias 0.0068 to 0.0121, margin 0.0072, SE ratio 0.9671 | [`RM18-one-sided-bias`](../../roadmap.md#red-cell-owners) |
| [multi-arm point-treatment DR-TMLE](multi-arm-dr-tmle.md) | property | `double_robust_contraction/outcome_correct_n4000` | positive | its own rule | coverage 0.8988 to 0.9542, bias 0.000900 | [`RM18-boundary`](../../roadmap.md#red-cell-owners) |
| [multi-arm point-treatment DR-TMLE](multi-arm-dr-tmle.md) | property | `double_robustness/treatment_correct` | positive | its own rule | bias 0.0015 to 0.0072, margin 0.0068, SE ratio 0.9998 | [`RM18-one-sided-bias`](../../roadmap.md#red-cell-owners) |
| [multi-arm point-treatment DR-TMLE](multi-arm-dr-tmle.md) | property | `interval_calibration/correctly_specified` | positive | its own rule | coverage 0.9300 to 0.9597, SE ratio 0.9258 to 1.0123 | [`RM18-boundary`](../../roadmap.md#red-cell-owners) |
| [multi-arm point-treatment DR-TMLE](multi-arm-dr-tmle.md) | property | `root_n_and_efficiency/n_500` | positive | its own rule | bias -0.000902, coverage 0.8965 to 0.9627, SE ratio 0.9888 | [`RM18-boundary`](../../roadmap.md#red-cell-owners) |
| [randomized multi-arm missing-outcome DR-TMLE](randomized-multi-arm-missing-outcome-dr-tmle.md) | property | `interval_calibration/ate__correctly_specified` | positive | its own rule | coverage 0.9287 to 0.9537, SE ratio 0.9294 to 1.0000, empirical efficiency ratio 0.9994 to 1.0751, reported efficiency ratio 0.9980 to 1.0008 | [`F4-calibration-draws`](../../roadmap.md#red-cell-owners) |
| [randomized multi-arm missing-outcome DR-TMLE](randomized-multi-arm-missing-outcome-dr-tmle.md) | property | `simultaneous_coverage/arms__simultaneous_band` | positive | its own rule | joint coverage 0.9196 to 0.9462 | [`F4-calibration-draws`](../../roadmap.md#red-cell-owners) |
| [observational missing-data DR-TMLE with a composite indicator](observational-missing-data-dr-tmle.md) | truth | `cleverly-composite-drtmle/three_arm_mar_outcome_and_treatment/ey[high]` | cleverly-composite-drtmle | its truth gate | bias -0.0054 to 0.0013, coverage 0.9225, SE ratio 0.9342 | [`composite-high-arm`](../../roadmap.md#red-cell-owners) |
| [observational missing-data DR-TMLE with a composite indicator](observational-missing-data-dr-tmle.md) | truth | `drtmle-r-composite/three_arm_mar_outcome_and_treatment/ey[high]` | drtmle-r-composite | its truth gate | bias -0.0054 to 0.0013, coverage 0.9237, SE ratio 0.9349 | [`composite-high-arm`](../../roadmap.md#red-cell-owners) |
| [CV-TMLE of the fold-local learned-rule value at exceptional and weak-blip laws](learned-rule-cvtmle-boundary.md) | truth | `cleverly-learned-rule-cvtmle/exceptional/ey_learned_rule[learned rule]` | cleverly-learned-rule-cvtmle | its truth gate | bias -0.000748 to 0.000535, coverage 0.8938, SE ratio 0.8211 | [`F27`](../../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) |
| [CV-TMLE of the fold-local learned-rule value at exceptional and weak-blip laws](learned-rule-cvtmle-boundary.md) | truth | `cleverly-learned-rule-cvtmle/weak_blip/ey_learned_rule[learned rule]` | cleverly-learned-rule-cvtmle | its truth gate | bias -0.000514 to 0.000732, coverage 0.8948, SE ratio 0.8314 | [`F27`](../../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) |
| [ordinary continuous modified treatment policies](continuous-modified-treatment-policies.md) | paired | `continuous_modified_policy/ate_policy[+0.25 vs natural course]` | paired | inconclusive: calibration leg | difference 0.000061 to 0.000361 within 0.000770, RMSE ratio bound 1.0587 vs 1.1000, coverage difference bound -0.0050 vs -0.0250, calibration excess bound 0.0648 vs 0.0500, resolution 0.0439 | [`RM18-comparator-density`](../../roadmap.md#red-cell-owners) |
| [ordinary point modified treatment policies](point-modified-treatment-policies.md) | property | `interval_calibration/halve__correctly_specified` | positive | its own rule | coverage 0.9479 to 0.9709, SE ratio 1.0011 to 1.0881, empirical efficiency ratio 0.9208 to 1.0003, reported efficiency ratio 1.0003 to 1.0034 | [`mtp-point-calibration`](../../roadmap.md#red-cell-owners) |
| [ordinary weighted end-of-study longitudinal TMLE](ordinary-weighted-end-of-study-longitudinal-tmle.md) | property | `interval_calibration/static__correctly_specified` | positive | its own rule | coverage 0.9187 to 0.9454, SE ratio 0.9263 to 0.9985, empirical efficiency ratio 0.9855 to 1.0615, reported efficiency ratio 0.9783 to 0.9899 | [`RM18-ordinary-weighted`](../../roadmap.md#red-cell-owners) |
| [ordinary weighted end-of-study longitudinal TMLE](ordinary-weighted-end-of-study-longitudinal-tmle.md) | property | `targeting_necessity/dynamic__targeted` | positive | the family's joint clause | bias -0.0025 to 0.0019, margin 0.0075 | [`RM18-ordinary-weighted`](../../roadmap.md#red-cell-owners) |
| [ordinary weighted end-of-study longitudinal TMLE](ordinary-weighted-end-of-study-longitudinal-tmle.md) | property | `targeting_necessity/dynamic__untargeted` | control | the family's joint clause | bias 0.0229 to 0.0284, margin 0.0093 | [`RM18-ordinary-weighted`](../../roadmap.md#red-cell-owners) |
| [ordinary weighted end-of-study longitudinal TMLE](ordinary-weighted-end-of-study-longitudinal-tmle.md) | property | `targeting_necessity/static__targeted` | positive | the family's joint clause | bias -0.0027 to 0.0075, margin 0.0172 | [`RM18-ordinary-weighted`](../../roadmap.md#red-cell-owners) |
| [ordinary weighted end-of-study longitudinal TMLE](ordinary-weighted-end-of-study-longitudinal-tmle.md) | property | `targeting_necessity/static__untargeted` | control | its own rule | bias -0.0242 to -0.0148, margin 0.0158 | [`RM18-ordinary-weighted`](../../roadmap.md#red-cell-owners) |
| [ordinary weighted end-of-study longitudinal TMLE](ordinary-weighted-end-of-study-longitudinal-tmle.md) | property | `type_i_error/static__sharp_null` | positive | its own rule | rejection 0.0750, 0.0530 to 0.1022 | [`RM18-ordinary-weighted`](../../roadmap.md#red-cell-owners) |
| [ordinary known stochastic categorical longitudinal TMLE](stochastic-categorical-longitudinal-tmle.md) | property | `power/mix__alternative` | positive | its own rule | rejection 0.5575, 0.5114 to 0.6029 | [`F1-power-design`](../../roadmap.md#red-cell-owners) |
| [longitudinal modified treatment policies](longitudinal-modified-treatment-policies.md) | property | `crossfit_overfitting/cross_fitted_mtp_ltmle` | positive | the family's joint clause | SE ratio 1.1463 to 1.1877 | [`mtp-longitudinal-limits`](../../roadmap.md#red-cell-owners) |
| [longitudinal modified treatment policies](longitudinal-modified-treatment-policies.md) | property | `crossfit_overfitting/in_sample_control` | control | its own rule | SE ratio 0.7276 to 0.7543 | [`mtp-longitudinal-limits`](../../roadmap.md#red-cell-owners) |
| [longitudinal modified treatment policies](longitudinal-modified-treatment-policies.md) | property | `interval_calibration/categorical_mtp__correctly_specified` | positive | its own rule | coverage 0.9289 to 0.9560, SE ratio 0.9432 to 1.0214, empirical efficiency ratio 1.0746 to 1.1643, reported efficiency ratio 1.0948 to 1.1006 | [`mtp-longitudinal-limits`](../../roadmap.md#red-cell-owners) |
| [cross-fitted weighted end-of-study longitudinal TMLE](cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | paired | `selected_censored_end_of_study/ey_regimen[always]` | paired | underpowered: coverage leg, calibration leg, calibration resolution | difference -0.000150 to 0.000202 within 0.0031, RMSE ratio bound 1.0162 vs 1.1000, coverage difference bound -0.0475 vs -0.0250, calibration excess bound 0.1050 vs 0.0500, resolution 0.1121 | [`RM18-boundary`](../../roadmap.md#red-cell-owners) |
| [cross-fitted weighted end-of-study longitudinal TMLE](cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | paired | `selected_censored_end_of_study/ey_regimen[never]` | paired | underpowered: coverage leg, calibration resolution | difference -0.000350 to 0.000263 within 0.0047, RMSE ratio bound 1.0245 vs 1.1000, coverage difference bound -0.0300 vs -0.0250, calibration excess bound 0.0465 vs 0.0500, resolution 0.0662 | [`RM18-fixed-weights`](../../roadmap.md#red-cell-owners) |
| [cross-fitted weighted end-of-study longitudinal TMLE](cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | paired | `selected_censored_end_of_study/ey_regimen[treat then continue if l2 positive]` | paired | underpowered: coverage leg, calibration resolution | difference -0.000118 to 0.000400 within 0.0032, RMSE ratio bound 1.0175 vs 1.1000, coverage difference bound -0.0425 vs -0.0250, calibration excess bound 0.0382 vs 0.0500, resolution 0.1119 | [`RM18-boundary`](../../roadmap.md#red-cell-owners) |
| [cross-fitted weighted end-of-study longitudinal TMLE](cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | property | `double_robustness/static__both_wrong` | control | its own rule | bias -0.0251 to -0.0151, margin 0.0167, SE ratio 0.6640 | [`RM18-boundary`](../../roadmap.md#red-cell-owners) |
| [cross-fitted weighted end-of-study longitudinal TMLE](cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | property | `interval_calibration/static__correctly_specified` | positive | its own rule | coverage 0.9232 to 0.9492, SE ratio 0.9430 to 1.0203, empirical efficiency ratio 1.0195 to 1.1006, reported efficiency ratio 1.0333 to 1.0442 | [`RM18-fixed-weights`](../../roadmap.md#red-cell-owners) |
| [ordinary survival-curve longitudinal TMLE](ordinary-survival-curve-longitudinal-tmle.md) | property | `simultaneous_coverage/all_reported__simultaneous_band` | positive | its own rule | joint coverage 0.9140 to 0.9283 | [`band-finite-sample`](../../roadmap.md#red-cell-owners) |
| [default simultaneous bands across shipped fit shapes](default-simultaneous-bands.md) | property | `simultaneous_coverage/categorical_ltmle__simultaneous_band` | positive | its own rule | joint coverage 0.9101 to 0.9382 | [`band-finite-sample`](../../roadmap.md#red-cell-owners) |
| [default simultaneous bands across shipped fit shapes](default-simultaneous-bands.md) | property | `simultaneous_coverage/categorical_ltmle_crossfit__simultaneous_band` | positive | its own rule | joint coverage 0.9173 to 0.9443 | [`band-finite-sample`](../../roadmap.md#red-cell-owners) |
| [default simultaneous bands across shipped fit shapes](default-simultaneous-bands.md) | property | `simultaneous_coverage/cde_z0__simultaneous_band` | positive | its own rule | joint coverage 0.9015 to 0.9309 | [`band-finite-sample`](../../roadmap.md#red-cell-owners) |
| [full-refit bootstrap and derived contrasts](full-refit-bootstrap-and-derived-contrasts.md) | property | `interval_calibration/boot_ate_clustered__correctly_specified` | positive | its own rule | coverage 0.9123 to 0.9342, SE ratio 0.9496 to 1.0065 | [`X20-bootstrap`](../../roadmap.md#red-cell-owners) |
| [full-refit bootstrap and derived contrasts](full-refit-bootstrap-and-derived-contrasts.md) | property | `interval_calibration/boot_ate_crossfit__correctly_specified` | positive | its own rule | coverage 0.9741 to 0.9857, SE ratio 1.2019 to 1.3336 | [`X20-bootstrap`](../../roadmap.md#red-cell-owners) |
| [full-refit bootstrap and derived contrasts](full-refit-bootstrap-and-derived-contrasts.md) | property | `interval_calibration/boot_ey_crossfit__correctly_specified` | positive | its own rule | coverage 0.9730 to 0.9848, SE ratio 1.1663 to 1.3066 | [`X20-bootstrap`](../../roadmap.md#red-cell-owners) |
<!-- /generated -->

## Reporting studies with no red row

These studies publish under the `reporting` policy, and none of them has a red row. The policy
check reads in one direction. A red row needs `reporting`, but `reporting` does not need a red
row. A move to `gated` is a separate registry decision.

<!-- generated: reporting-without-red -->
| study | verdicts it publishes | red rows |
| --- | --- | --- |
| [cross-fitted clustered longitudinal TMLE on a t reference at few clusters](few-cluster-cross-fitted-longitudinal-tmle.md) | 13 | 0 |
| [repeated point-treatment cross-fitted TMLE](repeated-cross-fitting.md) | 31 | 0 |
| [outcome-adaptive point-treatment C-TMLE](outcome-adaptive-point-treatment-c-tmle.md) | 29 | 0 |
| [cross-fitted end-of-study longitudinal TMLE](cross-fitted-end-of-study-longitudinal-tmle.md) | 47 | 0 |
| [omitted-variable bound standard error](omitted-variable-bound-standard-error.md) | 14 | 0 |
| [calibration-slope warning](calibration-slope-warning.md) | 13 | 0 |
<!-- /generated -->
