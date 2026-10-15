# Vector modified treatment policies

This study validates modified treatment policies on a vector treatment. A vector holds a binary
flag and one or two doses. The point `TMLE` fits it with `policies=`, and `LTMLE` fits it as a
node. Both read the ratio from the joint density that a `JointPolicy` induces
(`ratio="density"`) or from a stacked classifier (`ratio="classifier"`). The
[vector treatments](../point-treatment-tmle.md#vector-treatments) and
[vector nodes](../longitudinal-tmle.md#vector-nodes-with-a-continuous-component) sections state
the estimators, their conditions and their exact-law evidence.

**Status of this page.** The study is registered, and its Python phases have run. The 20 truth
tests and the 36 property cells pass (`performance-tests.csv`, `properties.csv`). The property
rows were fitted at commit `8b0e5b9b` on two machines.
[Provenance revisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/provenance-revisions.md)
records that run.
The evidence label is "Python property evidence, with comparator runs pending". The evidence has
two parts.

| part | what it is | status |
| --- | --- | --- |
| Python property evidence | exact laws with closed-form efficiency bounds, interval calibration with two controls, double robustness in four nuisance configurations, and three necessity pairs | complete: 20 of 20 truth tests and 36 of 36 property cells pass. It decides whether the method ships |
| `lmtp` comparator runs | the pinned `lmtp` 1.5.4 fits the same samples with vector `trt` nodes, and the study pairs the results | pending. The study record sets `pending_reference="lmtp"`, so no paired row is published. The runs register in a later change, and a finding in them does not block the method. The fast-tier witnesses and the Python property cells carry the claim until then |

The publication policy is `reporting`. A red cell stays red on this page and in the
[red-cell ledger](red-cells.md), with the owner row that names it.

## What was compared

The four scenarios draw 1,000 samples of 2,000 rows each.

| scenario | law and plans | pairing |
| --- | --- | --- |
| `point_vector_mixed` | the point law, `TMLE`, treatment $(F, D)$ with a binary flag and a dose. The plans are `natural course`, `dose up` and `flag scaled`. The ratio is the oracle induced density, which `lmtp` receives with both shifted components | margin-read: `TMLE` and `lmtp` target by different constructions |
| `point_vector_categorical` | the categorical law, `TMLE`, treatment $(A_1, A_2)$ with two binary components. The plans are `natural course`, `a2 off` and `a1 on`. The ratio is the discrete formula on the exact joint probabilities | margin-read |
| `longitudinal_vector_mixed` | the two-node law, `LTMLE` at one fold, nodes $(F_1, D_1)$ and $(F_2, D_2)$. The plans are the point plans at both nodes. Each node's ratio (`node_ratio`) and shifted components go to `lmtp` | exact under a gap of $10^{-6}$ |
| `longitudinal_vector_ordered_classifier` | the ordered law as one `LTMLE` node $(F, D_1, D_2)$. The plans are `natural course`, `ordered` and `full`. Each side fits its own stacked classifier | the parity rule below |

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets | 1,000 samples of 2,000 rows per scenario | the identical rows |
| density ratio | an oracle pooled-hazard density with `ceil(320 (n / 2000)^(2/3))` bins (320 at n = 2,000), and the exact joint probabilities on the categorical law | the same ratio array, read from the replicate data |
| outcome regressions | correctly specified logistic models: main terms plus the squared doses, or the saturated joint levels on the categorical law | `SL.glm` on the same columns |
| classifier of the ordered scenario | an unpenalized logistic regression of the stack label on one feature, the plan's true log ratio, which the learner computes from each stacked row | the R twin `SL.logratio.<plan>`, with `mtp = TRUE` |
| intervals | pointwise 95% Wald from the influence curve | the same, from `lmtp`'s influence curve |

The comparator adapter is `tests/canonical/lmtp_vector_adapter.R`. It builds an `LmtpTask` whose
treatment nodes are vectors, such as `trt = list(c("F", "D"))`. With `mtp = FALSE`, `lmtp` 1.5.4
multiplies each ratio by the indicator that no component moved (`R/density_ratios.R`, lines 58 to
61). That indicator is zero on every row that a plan moves, so the adapter sets `mtp = TRUE`.

The parity check requires that the share of positive `lmtp` ratios equal the package's share. Under
`mtp = TRUE` both shares are 1, because neither classifier predicts 0. Under `mtp = FALSE` the
`lmtp` share falls to the share of unmoved rows, and the check fails.

The ordered pair reads by three outcomes, declared before any run. Every ratio gap below $10^{-6}$
makes the pair an exactness witness. Every relative gap below $10^{-4}$, with a diagnosed and
named numerical cause, makes it a margin-read pair. Any other reading blocks the declaration until
the cause is found.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two nodes, each a binary flag and a truncated-normal dose | `ate_regimen[dose up vs natural course]` | difference in mean outcome between the plans "add 0.5 to each dose, capped at 5.5" against "leave the observed treatment mechanism unchanged" | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.000489 to 0.000894 | 0.9510 | 1.0294 | pass |
| two nodes, each a binary flag and a truncated-normal dose | `ate_regimen[flag scaled vs natural course]` | difference in mean outcome between the plans "set each flag to one, and multiply the dose by 1.25 where the natural flag is one" against "leave the observed treatment mechanism unchanged" | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.0017 to 0.0016 | 0.9510 | 0.9928 | pass |
| two nodes, each a binary flag and a truncated-normal dose | `ey_regimen[dose up]` | mean outcome under the plan add 0.5 to each dose, capped at 5.5 | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.000840 to 0.0015 | 0.9520 | 1.0018 | pass |
| two nodes, each a binary flag and a truncated-normal dose | `ey_regimen[flag scaled]` | mean outcome under the plan set each flag to one, and multiply the dose by 1.25 where the natural flag is one | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.0019 to 0.0020 | 0.9500 | 0.9572 | pass |
| two nodes, each a binary flag and a truncated-normal dose | `ey_regimen[natural course]` | mean outcome under the plan leave the observed treatment mechanism unchanged | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.000810 to 0.0010 | 0.9440 | 0.9994 | pass |
| one node of a binary flag and two ordered doses, its ratio a fitted stacked classifier | `ate_regimen[full vs natural course]` | difference in mean outcome between the plans "set the flag to one where the first dose is at least 3, scale the first dose where the flag is one, and move the second dose as in ordered" against "leave the observed treatment mechanism unchanged" | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.000951 to 0.000852 | 0.9430 | 0.9776 | pass |
| one node of a binary flag and two ordered doses, its ratio a fitted stacked classifier | `ate_regimen[ordered vs natural course]` | difference in mean outcome between the plans "multiply the first dose by 1.25, then add 0.5 to the second where the first is at most 3" against "leave the observed treatment mechanism unchanged" | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.000909 to 0.000863 | 0.9510 | 1.0058 | pass |
| one node of a binary flag and two ordered doses, its ratio a fitted stacked classifier | `ey_regimen[full]` | mean outcome under the plan set the flag to one where the first dose is at least 3, scale the first dose where the flag is one, and move the second dose as in ordered | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.0015 to 0.000988 | 0.9540 | 1.0021 | pass |
| one node of a binary flag and two ordered doses, its ratio a fitted stacked classifier | `ey_regimen[natural course]` | mean outcome under the plan leave the observed treatment mechanism unchanged | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.0011 to 0.000618 | 0.9660 | 1.0554 | pass |
| one node of a binary flag and two ordered doses, its ratio a fitted stacked classifier | `ey_regimen[ordered]` | mean outcome under the plan multiply the first dose by 1.25, then add 0.5 to the second where the first is at most 3 | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.0015 to 0.000984 | 0.9510 | 1.0332 | pass |
| a point treatment of two binary components | `ate_policy[a1 on vs natural course]` | difference in means under the modified treatment policies "set the first component to one" against "leave the observed treatment mechanism unchanged" | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.0011 to 0.000736 | 0.9580 | 1.0219 | pass |
| a point treatment of two binary components | `ate_policy[a2 off vs natural course]` | difference in means under the modified treatment policies "set the second component to zero where the first is one" against "leave the observed treatment mechanism unchanged" | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.0010 to 0.000573 | 0.9630 | 1.0436 | pass |
| a point treatment of two binary components | `ey_policy[a1 on]` | mean under the modified treatment policy set the first component to one | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.0020 to 0.000659 | 0.9560 | 1.0125 | pass |
| a point treatment of two binary components | `ey_policy[a2 off]` | mean under the modified treatment policy set the second component to zero where the first is one | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.0019 to 0.000513 | 0.9530 | 1.0340 | pass |
| a point treatment of two binary components | `ey_policy[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.0013 to 0.000438 | 0.9450 | 1.0094 | pass |
| a point treatment of one binary flag and one truncated-normal dose | `ate_policy[dose up vs natural course]` | difference in means under the modified treatment policies "add 0.5 to each dose, capped at 5.5" against "leave the observed treatment mechanism unchanged" | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.000560 to 0.000410 | 0.9520 | 1.0031 | pass |
| a point treatment of one binary flag and one truncated-normal dose | `ate_policy[flag scaled vs natural course]` | difference in means under the modified treatment policies "set each flag to one, and multiply the dose by 1.25 where the natural flag is one" against "leave the observed treatment mechanism unchanged" | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.0012 to 0.0010 | 0.9410 | 0.9825 | pass |
| a point treatment of one binary flag and one truncated-normal dose | `ey_policy[dose up]` | mean under the modified treatment policy add 0.5 to each dose, capped at 5.5 | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.0011 to 0.0010 | 0.9450 | 0.9889 | pass |
| a point treatment of one binary flag and one truncated-normal dose | `ey_policy[flag scaled]` | mean under the modified treatment policy set each flag to one, and multiply the dose by 1.25 where the natural flag is one | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.0015 to 0.0015 | 0.9320 | 0.9518 | pass |
| a point treatment of one binary flag and one truncated-normal dose | `ey_policy[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | `cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment | -0.000838 to 0.000958 | 0.9420 | 0.9938 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

Python property evidence, with comparator runs pending. The `lmtp` runs have not run, so this
study publishes no paired row, and `equivalence.csv` is empty. The change that registers the runs
adds the paired table here.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `longitudinal_vector_density__both_correct` | positive | the two-node flag-and-dose contrast, its ratios the oracle joint induced density: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0013 to 0.0011, margin 0.0051, SE ratio 0.9877 | pass |
| `double_robustness` | `longitudinal_vector_density__both_wrong` | control | the two-node flag-and-dose contrast, its ratios the oracle joint induced density: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0615 to 0.0644, margin 0.0062, SE ratio 1.4677 | pass |
| `double_robustness` | `longitudinal_vector_density__outcome_correct` | positive | the two-node flag-and-dose contrast, its ratios the oracle joint induced density: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000657 to 0.0022, margin 0.0061, SE ratio 1.4254 | pass |
| `double_robustness` | `longitudinal_vector_density__outcome_wrong` | positive | the two-node flag-and-dose contrast, its ratios the oracle joint induced density: the outcome regression is misspecified and the joint density is the oracle, read from 320 oracle bins per dose | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000841 to 0.0016, margin 0.0053, SE ratio 1.0045 | pass |
| `double_robustness` | `longitudinal_vector_ordered_density__both_correct` | positive | the ordered two-dose node contrast, its ratio the oracle joint induced density: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000267 to 0.000990, margin 0.0027, SE ratio 0.9983 | pass |
| `double_robustness` | `longitudinal_vector_ordered_density__both_wrong` | control | the ordered two-dose node contrast, its ratio the oracle joint induced density: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0483 to 0.0499, margin 0.0034, SE ratio 1.1535 | pass |
| `double_robustness` | `longitudinal_vector_ordered_density__outcome_correct` | positive | the ordered two-dose node contrast, its ratio the oracle joint induced density: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000550 to 0.000694, margin 0.0027, SE ratio 1.4222 | pass |
| `double_robustness` | `longitudinal_vector_ordered_density__outcome_wrong` | positive | the ordered two-dose node contrast, its ratio the oracle joint induced density: the outcome regression is misspecified and the joint density is the oracle, read from 320 oracle bins per dose | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000515 to 0.000796, margin 0.0028, SE ratio 0.9779 | pass |
| `double_robustness` | `point_vector_ordered_density__both_correct` | positive | the ordered two-dose contrast read as a point treatment, its ratio the oracle joint induced density: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000793 to 0.000461, margin 0.0027, SE ratio 0.9984 | pass |
| `double_robustness` | `point_vector_ordered_density__both_wrong` | control | the ordered two-dose contrast read as a point treatment, its ratio the oracle joint induced density: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0195 to 0.0206, margin 0.0024, SE ratio 1.6662 | pass |
| `double_robustness` | `point_vector_ordered_density__outcome_correct` | positive | the ordered two-dose contrast read as a point treatment, its ratio the oracle joint induced density: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000210 to 0.000754, margin 0.0021, SE ratio 1.8336 | pass |
| `double_robustness` | `point_vector_ordered_density__outcome_wrong` | positive | the ordered two-dose contrast read as a point treatment, its ratio the oracle joint induced density: the outcome regression is misspecified and the joint density is the oracle, read from 320 oracle bins per dose | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000550 to 0.000738, margin 0.0028, SE ratio 1.0138 | pass |
| `image_necessity` | `ordered_crossed__declared_image` | positive | the ordered-law node contrast of the crossed plan against the natural course: the joint density evaluates the flag map at each preimage's natural first dose, with the outcome misspecified | bias interval inside the equivalence margin | bias -0.000744 to 0.000720, margin 0.0022 | pass |
| `image_necessity` | `ordered_crossed__shifted_read_control` | control | the ordered-law node contrast of the crossed plan against the natural course: the same fit evaluates the flag map at the value's own first dose | bias interval must fall entirely outside the margin | bias 0.0105 to 0.0122, margin 0.0027 | pass |
| `interval_calibration` | `longitudinal_vector_classifier__correctly_specified` | positive | the two-node flag-and-dose contrast, its ratios a classifier on the true log ratio: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9439 to 0.9553, SE ratio 0.9806 to 1.0159, empirical efficiency ratio 0.9742 to 1.0093, reported efficiency ratio 0.9877 to 0.9915 | pass |
| `interval_calibration` | `longitudinal_vector_classifier__noise_control` | control | the two-node flag-and-dose contrast, its ratios a classifier on the true log ratio: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8164 to 0.8360, SE ratio 0.6842 to 0.7099, empirical efficiency ratio 1.3936 to 1.4462, reported efficiency ratio 0.9877 to 0.9915 | pass |
| `interval_calibration` | `longitudinal_vector_classifier__shrunken_se_control` | control | the two-node flag-and-dose contrast, its ratios a classifier on the true log ratio: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8152 to 0.8349, SE ratio 0.6864 to 0.7114, empirical efficiency ratio 0.9738 to 1.0094, reported efficiency ratio 0.6914 to 0.6941 | pass |
| `interval_calibration` | `longitudinal_vector_density__correctly_specified` | positive | the two-node flag-and-dose contrast, its ratios the oracle joint induced density: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9422 to 0.9540, SE ratio 0.9813 to 1.0196, empirical efficiency ratio 0.9753 to 1.0127, reported efficiency ratio 0.9919 to 0.9962 | pass |
| `interval_calibration` | `longitudinal_vector_density__noise_control` | control | the two-node flag-and-dose contrast, its ratios the oracle joint induced density: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8193 to 0.8392, SE ratio 0.6891 to 0.7158, empirical efficiency ratio 1.3885 to 1.4417, reported efficiency ratio 0.9919 to 0.9962 | pass |
| `interval_calibration` | `longitudinal_vector_density__shrunken_se_control` | control | the two-node flag-and-dose contrast, its ratios the oracle joint induced density: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8207 to 0.8405, SE ratio 0.6861 to 0.7136, empirical efficiency ratio 0.9750 to 1.0139, reported efficiency ratio 0.6943 to 0.6973 | pass |
| `interval_calibration` | `longitudinal_vector_ordered_classifier__correctly_specified` | positive | the ordered two-dose node contrast, its ratio a classifier on the true log ratio: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9423 to 0.9538, SE ratio 0.9771 to 1.0140, empirical efficiency ratio 0.9793 to 1.0157, reported efficiency ratio 0.9914 to 0.9941 | pass |
| `interval_calibration` | `longitudinal_vector_ordered_classifier__noise_control` | control | the ordered two-dose node contrast, its ratio a classifier on the true log ratio: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8194 to 0.8389, SE ratio 0.6878 to 0.7123, empirical efficiency ratio 1.3938 to 1.4435, reported efficiency ratio 0.9914 to 0.9941 | pass |
| `interval_calibration` | `longitudinal_vector_ordered_classifier__shrunken_se_control` | control | the ordered two-dose node contrast, its ratio a classifier on the true log ratio: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8204 to 0.8399, SE ratio 0.6843 to 0.7097, empirical efficiency ratio 0.9792 to 1.0154, reported efficiency ratio 0.6940 to 0.6959 | pass |
| `interval_calibration` | `longitudinal_vector_ordered_density__correctly_specified` | positive | the ordered two-dose node contrast, its ratio the oracle joint induced density: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9406 to 0.9491, SE ratio 0.9817 to 1.0073, empirical efficiency ratio 0.9900 to 1.0160, reported efficiency ratio 0.9964 to 0.9984 | pass |
| `interval_calibration` | `longitudinal_vector_ordered_density__noise_control` | control | the ordered two-dose node contrast, its ratio the oracle joint induced density: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8215 to 0.8355, SE ratio 0.6914 to 0.7101, empirical efficiency ratio 1.4047 to 1.4424, reported efficiency ratio 0.9964 to 0.9984 | pass |
| `interval_calibration` | `longitudinal_vector_ordered_density__shrunken_se_control` | control | the ordered two-dose node contrast, its ratio the oracle joint induced density: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8207 to 0.8348, SE ratio 0.6871 to 0.7053, empirical efficiency ratio 0.9900 to 1.0161, reported efficiency ratio 0.6975 to 0.6989 | pass |
| `interval_calibration` | `point_vector_classifier__correctly_specified` | positive | the flag-and-dose point contrast, its ratio a classifier on the true log ratio: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9418 to 0.9534, SE ratio 0.9711 to 1.0064, empirical efficiency ratio 0.9932 to 1.0293, reported efficiency ratio 0.9987 to 1.0006 | pass |
| `interval_calibration` | `point_vector_classifier__noise_control` | control | the flag-and-dose point contrast, its ratio a classifier on the true log ratio: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8192 to 0.8387, SE ratio 0.6845 to 0.7110, empirical efficiency ratio 1.4057 to 1.4603, reported efficiency ratio 0.9987 to 1.0007 | pass |
| `interval_calibration` | `point_vector_classifier__shrunken_se_control` | control | the flag-and-dose point contrast, its ratio a classifier on the true log ratio: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8165 to 0.8361, SE ratio 0.6793 to 0.7048, empirical efficiency ratio 0.9928 to 1.0301, reported efficiency ratio 0.6991 to 0.7005 | pass |
| `interval_calibration` | `point_vector_density__correctly_specified` | positive | the flag-and-dose point contrast, its ratio the oracle joint induced density: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9440 to 0.9514, SE ratio 0.9806 to 1.0041, empirical efficiency ratio 0.9961 to 1.0202, reported efficiency ratio 0.9998 to 1.0008 | pass |
| `interval_calibration` | `point_vector_density__noise_control` | control | the flag-and-dose point contrast, its ratio the oracle joint induced density: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8249 to 0.8374, SE ratio 0.6969 to 0.7132, empirical efficiency ratio 1.4025 to 1.4353, reported efficiency ratio 0.9998 to 1.0007 | pass |
| `interval_calibration` | `point_vector_density__shrunken_se_control` | control | the flag-and-dose point contrast, its ratio the oracle joint induced density: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8151 to 0.8279, SE ratio 0.6866 to 0.7023, empirical efficiency ratio 0.9970 to 1.0198, reported efficiency ratio 0.6998 to 0.7005 | pass |
| `jacobian_necessity` | `ordered_full__declared_jacobian` | positive | the ordered-law node contrast of the full plan against the natural course: the joint density multiplies each preimage by the inverse derivative of the D1 scale, the only non-unit Jacobian, with the outcome misspecified | bias interval inside the equivalence margin | bias -0.0010 to 0.000778, margin 0.0027 | pass |
| `jacobian_necessity` | `ordered_full__jacobian_drop_control` | control | the ordered-law node contrast of the full plan against the natural course: the same fit drops the inverse derivative of the D1 scale | bias interval must fall entirely outside the margin | bias 0.0045 to 0.0064, margin 0.0028 | pass |
| `preimage_necessity` | `flag_scaled__declared_preimage` | positive | the two-node contrast of the flag-scaled plan against the natural course: the joint density sums over every categorical preimage, with the outcome misspecified | bias interval inside the equivalence margin | bias -0.0010 to 0.0024, margin 0.0053 | pass |
| `preimage_necessity` | `flag_scaled__preimage_cut_control` | control | the two-node contrast of the flag-scaled plan against the natural course: the same fit keeps only the preimages in the value's own categorical cell | bias interval must fall entirely outside the margin | bias 0.0758 to 0.0806, margin 0.0073 | pass |
<!-- /generated -->

The truths and the efficiency bounds come from the exact laws and their closed forms
(`tests/studies/vector_mtp_common.py`). The quadrature is checked at $10^{-10}$.

| family | cells | replications and size |
| --- | --- | --- |
| `interval_calibration` | six positives: point `TMLE`, two-node `LTMLE` and the ordered law as one `LTMLE` node, each on the density route and on the classifier route. Each positive has a shrunken-SE control and a noise control | 24,000, 9,600 and 19,200 replications on the density routes, and 10,000 on each classifier route, at n = 2,000. The design test sizes them |
| `double_robustness` | point `TMLE` on the ordered law under `full`, two-node `LTMLE` under `flag scaled`, and the ordered law as one `LTMLE` node under `full`. Each runs `both_correct`, `outcome_correct`, `outcome_wrong` and the `both_wrong` control | 2,000 replications at n = 2,000 |
| `preimage_necessity` | two-node `LTMLE` under `flag scaled`, wrong outcome regression. The declared fit against the same draw with the preimage sum cut to the observed categorical cell | 1,000 pairs at n = 2,000 |
| `jacobian_necessity` | the ordered law under `full`, wrong outcome regression. The declared fit against the same draw with the Jacobian of the `D1` scale dropped. That scale is the only map whose Jacobian is not 1 | 1,000 pairs at n = 2,000 |
| `image_necessity` | the ordered law under `crossed`. The declared fit against the same draw with the flag map read at the observed second value of $D_1$ instead of the natural value of the preimage | 1,000 pairs at n = 2,000 |

Each necessity control removes one term that can vanish at the truth, so each needs a nonzero
witness. With a wrong outcome regression the targeting step carries the whole correction, and each
defect then moves the estimate. The `crossed` plan exists because the `full` plan is blind to the
flag map's read of $D_1$. The exact-law shift between the declared and the defective map is 0.000
under `full` and 1.243 SD under `crossed`
(`tests/unit/test_vector_mtp_design.py`). A control passes only when its pair moves by more than
the declared displacement.

The design test sizes every replication count before the run. It requires the product of the pass
probabilities of the six calibration positives and their controls to reach 0.90. The product holds
with every probed number moved one Monte Carlo standard error toward failing.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_passed` | 20 | truth tests passing |
| `independent_tests_total` | 20 | truth tests reported |
| `paired_tests_passed` | 0 | paired comparisons passing |
| `paired_tests_total` | 0 | paired comparisons reported |
| `property_cells_passed` | 36 | property cells passing |
| `property_cells_total` | 36 | property cells reported |
| `max_standardized_bias` | 0.0464 | largest primary standardized bias |
| `min_coverage` | 0.9320 | lowest primary coverage |
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
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:necessity_displacement` | 0.1000 | minimum Jacobian and image necessity displacement |
| `margin:preimage_displacement` | 0.1000 | minimum preimage necessity displacement |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 10 | the same bound, in primary replications |

## Limits

| limit | what it means for use |
| --- | --- |
| three laws with a binary flag and at most two doses, at n = 2,000 | the evidence covers these laws and this size. The study has no size ladder, so it claims no rate |
| the density is an oracle | the density route reads the generating pooled-hazard probabilities at a bin count that grows as $n^{2/3}$. The default estimated density is not tested here |
| the classifier is a one-feature oracle | the classifier route reads the correctly specified logistic model of the plan's log ratio. The study says nothing about the rate or the interval of a learned classifier |
| the density of $D_2$ and the map of $D_2$ may read the shifted $D_1$ without a failing cell | those defects move the outcome-wrong positives by 0.42 and -0.19 SD only. The fast-tier mutation witnesses in `tests/unit/test_vector_density_mutations.py` carry them |
| the point pairs compare two constructions | `TMLE` targets with the covariate submodel over the observed and the policy values, and `lmtp` with a ratio-weighted intercept. The pairs read under the default margins and are not exactness checks |
| the `lmtp` comparison is pending | the agreement table is empty until the reference runs register. The Python cells carry the claim until then |
| a policy must be known and fixed | a policy learned from the sample, or one that depends on the law, is outside this evidence |
