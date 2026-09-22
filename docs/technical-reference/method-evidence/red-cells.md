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

The ledger reads verdict rows only. Four study modules also define a `scientific_failures` hook.
The regeneration adds each hook's rows to the failures it gates or reports. These hooks audit
fits and publish no verdict cell, so the ledger does not read them.

| study | policy | what the hook reports | what holds its result |
| --- | --- | --- | --- |
| [DR-TMLE for binary complete data](canonical-dr-tmle.md) | `reporting` | the score audit of both implementations, and the solver flag of `cleverly` | the study page and `fit-diagnostics.csv` |
| [multi-arm point-treatment DR-TMLE](multi-arm-dr-tmle.md) | `reporting` | the same two audits | the study page and `fit-diagnostics.csv` |
| [ordinary missing-outcome natural-course TMLE](ordinary-missing-outcome-natural-course-tmle.md) | `gated` | the exact-equality probe of the scale workaround | `scale-probe.csv`. The test requires every row to pass |
| [stacked arm-indexed missing-outcome CV-TMLE](stacked-arm-indexed-missing-outcome-cvtmle.md) | `gated` | the same probe | `scale-probe.csv`. The test requires every row to pass |

`tests/unit/test_red_cell_ledger.py` fails when a study module defines a hook that this table
does not list.

## Who owns each red row

Each owner is an `id` in the
["What this row asks for"](../../roadmap.md#what-this-row-asks-for) table of RM18.
[F18](../../roadmap.md#f18-selector-path-c-tmle-inference) and
[F19](../../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) wait on a
published result, or on a natural extension that meets the
[Eligibility](../../roadmap.md#eligibility) conditions. Each `RM18-` owner is a follow-up that
RM18 records. Its row states whether its declared design has run.

A red row stays red under the `reporting` policy. It is read again only under the condition its owner names. The ledger moves
no margin and changes no verdict.

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
| [`RM18-fixed-weights`](../../roadmap.md#what-this-row-asks-for) | 2 | cross-fitted weighted end-of-study longitudinal TMLE |
| [`RM18-one-sided-bias`](../../roadmap.md#what-this-row-asks-for) | 3 | DR-TMLE for binary complete data, multi-arm point-treatment DR-TMLE |
| [`RM18-slopes`](../../roadmap.md#what-this-row-asks-for) | 2 | multi-arm point-treatment DR-TMLE |
| [`RM18-ordinary-weighted`](../../roadmap.md#what-this-row-asks-for) | 6 | ordinary weighted end-of-study longitudinal TMLE |
| [`RM18-boundary`](../../roadmap.md#what-this-row-asks-for) | 8 | selector-based multi-arm C-TMLE, DR-TMLE for binary complete data, multi-arm point-treatment DR-TMLE, cross-fitted weighted end-of-study longitudinal TMLE |
| [`RM18-comparator-density`](../../roadmap.md#what-this-row-asks-for) | 1 | ordinary continuous modified treatment policies |
| total | 33 | 8 studies |
<!-- /generated -->

## The red rows

The "measured" column gives the endpoints that the failed verdict reads. A property row uses the
same text as the "measured" column of its study page.

<!-- generated: red-cells -->
| study | kind | row | role | fails by | measured | owner |
| --- | --- | --- | --- | --- | --- | --- |
| [selector-based point-treatment C-TMLE](selector-based-point-treatment-c-tmle.md) | property | `selector_necessity/collaborative` | positive | its own rule | bias 0.0015 to 0.0037, margin 0.0030, RMSE ratio 0.2410 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based point-treatment C-TMLE](selector-based-point-treatment-c-tmle.md) | property | `selector_necessity/empty_control` | control | the family's joint clause | bias 0.0495 to 0.0511, margin 0.0022, RMSE ratio 0.2410 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based point-treatment C-TMLE](selector-based-point-treatment-c-tmle.md) | property | `type_i_error/sharp_null` | positive | its own rule | rejection 0.0700, 0.0412 to 0.1095 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `interval_calibration/correctly_specified` | positive | its own rule | coverage 0.9119 to 0.9454, SE ratio 0.8987 to 0.9862 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `root_n_and_efficiency/n_500` | positive | its own rule | bias 0.0010, coverage 0.8965 to 0.9627, SE ratio 0.9892 | [`RM18-boundary`](../../roadmap.md#what-this-row-asks-for) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `selector_necessity/discrete` | positive | its own rule | bias 0.1589 to 0.1660, margin 0.0069, RMSE ratio 1 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `selector_necessity/empty_control` | control | the family's joint clause | bias 0.1589 to 0.1660, margin 0.0069, RMSE ratio 1 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `selector_necessity/greedy` | positive | its own rule | bias 0.0279 to 0.0443, margin 0.0158, RMSE ratio 0.4415 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `selector_necessity/ordered` | positive | its own rule | bias 0.0154 to 0.0304, margin 0.0145, RMSE ratio 0.3781 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [selector-based multi-arm C-TMLE](selector-based-multi-arm-c-tmle.md) | property | `type_i_error/sharp_null` | positive | its own rule | rejection 0.0625, 0.0354 to 0.1004 | [`F18`](../../roadmap.md#f18-selector-path-c-tmle-inference) |
| [outcome-adaptive multi-arm C-TMLE](outcome-adaptive-multi-arm-c-tmle.md) | property | `generated_design/estimated` | positive | its own rule | coverage 0.9267 to 0.9677, SE ratio 0.9694 to 1.1026 | [`F19`](../../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| [outcome-adaptive multi-arm C-TMLE](outcome-adaptive-multi-arm-c-tmle.md) | property | `generated_design/oracle_design` | positive | its own rule | coverage 0.9282 to 0.9688, SE ratio 0.9797 to 1.1151 | [`F19`](../../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| [DR-TMLE for binary complete data](canonical-dr-tmle.md) | property | `double_robust_contraction/treatment_correct_n1500` | positive | its own rule | coverage 0.8864 to 0.9385, bias 0.0115 | [`RM18-boundary`](../../roadmap.md#what-this-row-asks-for) |
| [DR-TMLE for binary complete data](canonical-dr-tmle.md) | property | `double_robustness/outcome_correct` | positive | its own rule | bias 0.0026 to 0.0075, margin 0.0067, SE ratio 0.9888 | [`RM18-one-sided-bias`](../../roadmap.md#what-this-row-asks-for) |
| [DR-TMLE for binary complete data](canonical-dr-tmle.md) | property | `double_robustness/treatment_correct` | positive | its own rule | bias 0.0068 to 0.0121, margin 0.0072, SE ratio 0.9671 | [`RM18-one-sided-bias`](../../roadmap.md#what-this-row-asks-for) |
| [multi-arm point-treatment DR-TMLE](multi-arm-dr-tmle.md) | property | `double_robust_contraction/outcome_correct_n4000` | positive | its own rule | coverage 0.8988 to 0.9542, bias 0.000900 | [`RM18-boundary`](../../roadmap.md#what-this-row-asks-for) |
| [multi-arm point-treatment DR-TMLE](multi-arm-dr-tmle.md) | property | `double_robust_contraction/rate_outcome_correct` | positive | its own rule | slope -1.2342 to 3.7735 | [`RM18-slopes`](../../roadmap.md#what-this-row-asks-for) |
| [multi-arm point-treatment DR-TMLE](multi-arm-dr-tmle.md) | property | `double_robust_contraction/rate_treatment_correct` | positive | its own rule | slope -3.2412 to 3.3387 | [`RM18-slopes`](../../roadmap.md#what-this-row-asks-for) |
| [multi-arm point-treatment DR-TMLE](multi-arm-dr-tmle.md) | property | `double_robustness/treatment_correct` | positive | its own rule | bias 0.0015 to 0.0072, margin 0.0068, SE ratio 0.9998 | [`RM18-one-sided-bias`](../../roadmap.md#what-this-row-asks-for) |
| [multi-arm point-treatment DR-TMLE](multi-arm-dr-tmle.md) | property | `interval_calibration/correctly_specified` | positive | its own rule | coverage 0.9300 to 0.9597, SE ratio 0.9258 to 1.0123 | [`RM18-boundary`](../../roadmap.md#what-this-row-asks-for) |
| [multi-arm point-treatment DR-TMLE](multi-arm-dr-tmle.md) | property | `root_n_and_efficiency/n_500` | positive | its own rule | bias -0.000902, coverage 0.8965 to 0.9627, SE ratio 0.9888 | [`RM18-boundary`](../../roadmap.md#what-this-row-asks-for) |
| [ordinary continuous modified treatment policies](continuous-modified-treatment-policies.md) | paired | `continuous_modified_policy/ate_shift[+0.25 vs natural course]` | paired | inconclusive: calibration leg | difference 0.000061 to 0.000361 within 0.000770, RMSE ratio bound 1.0583 vs 1.1000, coverage difference bound -0.0063 vs -0.0250, calibration excess bound 0.0659 vs 0.0500, resolution 0.0450 | [`RM18-comparator-density`](../../roadmap.md#what-this-row-asks-for) |
| [ordinary weighted end-of-study longitudinal TMLE](ordinary-weighted-end-of-study-longitudinal-tmle.md) | property | `interval_calibration/static__correctly_specified` | positive | its own rule | coverage 0.9187 to 0.9454, SE ratio 0.9263 to 0.9985, empirical efficiency ratio 0.9855 to 1.0615, reported efficiency ratio 0.9783 to 0.9899 | [`RM18-ordinary-weighted`](../../roadmap.md#what-this-row-asks-for) |
| [ordinary weighted end-of-study longitudinal TMLE](ordinary-weighted-end-of-study-longitudinal-tmle.md) | property | `targeting_necessity/dynamic__targeted` | positive | the family's joint clause | bias -0.0025 to 0.0019, margin 0.0075 | [`RM18-ordinary-weighted`](../../roadmap.md#what-this-row-asks-for) |
| [ordinary weighted end-of-study longitudinal TMLE](ordinary-weighted-end-of-study-longitudinal-tmle.md) | property | `targeting_necessity/dynamic__untargeted` | control | the family's joint clause | bias 0.0229 to 0.0284, margin 0.0093 | [`RM18-ordinary-weighted`](../../roadmap.md#what-this-row-asks-for) |
| [ordinary weighted end-of-study longitudinal TMLE](ordinary-weighted-end-of-study-longitudinal-tmle.md) | property | `targeting_necessity/static__targeted` | positive | the family's joint clause | bias -0.0027 to 0.0075, margin 0.0172 | [`RM18-ordinary-weighted`](../../roadmap.md#what-this-row-asks-for) |
| [ordinary weighted end-of-study longitudinal TMLE](ordinary-weighted-end-of-study-longitudinal-tmle.md) | property | `targeting_necessity/static__untargeted` | control | its own rule | bias -0.0242 to -0.0148, margin 0.0158 | [`RM18-ordinary-weighted`](../../roadmap.md#what-this-row-asks-for) |
| [ordinary weighted end-of-study longitudinal TMLE](ordinary-weighted-end-of-study-longitudinal-tmle.md) | property | `type_i_error/static__sharp_null` | positive | its own rule | rejection 0.0750, 0.0530 to 0.1022 | [`RM18-ordinary-weighted`](../../roadmap.md#what-this-row-asks-for) |
| [cross-fitted weighted end-of-study longitudinal TMLE](cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | paired | `selected_censored_end_of_study/ey_regimen[always]` | paired | underpowered: coverage leg, calibration leg, calibration resolution | difference -0.000150 to 0.000202 within 0.0031, RMSE ratio bound 1.0162 vs 1.1000, coverage difference bound -0.0475 vs -0.0250, calibration excess bound 0.1050 vs 0.0500, resolution 0.1121 | [`RM18-boundary`](../../roadmap.md#what-this-row-asks-for) |
| [cross-fitted weighted end-of-study longitudinal TMLE](cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | paired | `selected_censored_end_of_study/ey_regimen[never]` | paired | underpowered: coverage leg, calibration resolution | difference -0.000350 to 0.000263 within 0.0047, RMSE ratio bound 1.0245 vs 1.1000, coverage difference bound -0.0300 vs -0.0250, calibration excess bound 0.0465 vs 0.0500, resolution 0.0662 | [`RM18-fixed-weights`](../../roadmap.md#what-this-row-asks-for) |
| [cross-fitted weighted end-of-study longitudinal TMLE](cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | paired | `selected_censored_end_of_study/ey_regimen[treat then continue if l2 positive]` | paired | underpowered: coverage leg, calibration resolution | difference -0.000118 to 0.000400 within 0.0032, RMSE ratio bound 1.0175 vs 1.1000, coverage difference bound -0.0425 vs -0.0250, calibration excess bound 0.0382 vs 0.0500, resolution 0.1119 | [`RM18-boundary`](../../roadmap.md#what-this-row-asks-for) |
| [cross-fitted weighted end-of-study longitudinal TMLE](cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | property | `double_robustness/static__both_wrong` | control | its own rule | bias -0.0251 to -0.0151, margin 0.0167, SE ratio 0.6640 | [`RM18-boundary`](../../roadmap.md#what-this-row-asks-for) |
| [cross-fitted weighted end-of-study longitudinal TMLE](cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | property | `interval_calibration/static__correctly_specified` | positive | its own rule | coverage 0.9232 to 0.9492, SE ratio 0.9430 to 1.0203, empirical efficiency ratio 1.0195 to 1.1006, reported efficiency ratio 1.0333 to 1.0442 | [`RM18-fixed-weights`](../../roadmap.md#what-this-row-asks-for) |
<!-- /generated -->

## Reporting studies with no red row

These studies publish under the `reporting` policy, and none of them has a red row. The policy
check reads in one direction. A red row needs `reporting`, but `reporting` does not need a red
row. A move to `gated` is a separate registry decision.

<!-- generated: reporting-without-red -->
| study | verdicts it publishes | red rows |
| --- | --- | --- |
| [repeated point-treatment cross-fitted TMLE](repeated-cross-fitting.md) | 31 | 0 |
| [outcome-adaptive point-treatment C-TMLE](outcome-adaptive-point-treatment-c-tmle.md) | 29 | 0 |
| [cross-fitted end-of-study longitudinal TMLE](cross-fitted-end-of-study-longitudinal-tmle.md) | 47 | 0 |
<!-- /generated -->
