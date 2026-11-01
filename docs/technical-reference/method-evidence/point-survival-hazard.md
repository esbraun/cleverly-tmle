# Point survival hazard

This study validates longitudinal TMLE with `survival_model="hazard"` and `targeting="iterative"`
on a held design. The design has one baseline treatment, held over every visit, and it reads one
follow-up time and one event code per unit (`TimeToEvent`). Each reported horizon targets the
discrete cause-specific hazards of every regimen and cause on its own (Moore and van der Laan
2009, Section 4.1). The section
[hazard-based survival targeting](../longitudinal-tmle.md#hazard-based-survival-targeting) states
the estimator and its exact-law evidence.

The laws are the `survival` and `competing` scenarios of
[point-treatment survival](point-treatment-survival.md). Every truth is an exact finite sum. Every
efficiency bound is the exact standard deviation of the efficient influence function over the
law's support.

The study pairs each fit with R `survtmle` 1.1.1, `method = "hazard"` (Benkeser, Carone and
Gilbert 2018), on the same samples. The evidence label is "comparator-backed". The `survtmle` rows
were fitted at commit `3828b2a8` with 1,600 samples per scenario. The `cleverly` estimates of that
run match the earlier Python-only run of the same samples to within $4 \times 10^{-14}$, with
every coverage indicator identical.

## What was compared

| setting | `cleverly` | R `survtmle` |
| --- | --- | --- |
| datasets | 1,600 samples of 2,000 rows for each scenario | the identical rows |
| scenarios | `survival`: five visits, the risks of both arms and their difference at visits 1, 3 and 5. `competing`: four visits and two causes, the incidences and their differences at visits 2 and 4 | the same |
| treatment model | a logistic regression on `W1 + W2` | `glm.trt = "W1 + W2"` |
| censoring model | a logistic regression on the arm, `W1` and `W2` at each visit | one pooled `glm` with one intercept and three slopes for each visit, which factorises by visit into the same model |
| hazard model | an unpenalized logistic regression on `W1`, `W2` and the arm at each node and cause | one pooled `glm` with the same terms for each visit and cause |
| targeting | `targeting="iterative"`, all causes fluctuated together in a multinomial logistic submodel, stopped at a relative score of 1e-10 | one cause at a time in a logistic submodel, stopped at `max |meanIC| <= 1/n` or at `maxIter = 100` |
| intervals | pointwise 95% Wald from the influence curve | the same |

The paired rows read under the default margins, because the two targeting loops differ by
construction. The runner declares `tol = 1/n` and `maxIter = 100`. The `survtmle` default
`tol = 1/sqrt(n)` stops before any targeting step at the laws of this study.

None of the 8,000 `survtmle` calls (1,600 samples, two scenarios, five horizons) failed. In 199
calls (2.5%) the loop stopped at `maxIter` with `max |meanIC|` above `1/n`, against about 4%
declared from the smoke run. Those fits stay in the paired rows, as the manifest's pairing rule
says. The run log of Study queue row 20 in `TRACKER.md` records the counts. The only other R
messages were `glm` rank-deficiency warnings and `speedglm` falling back to `glm`.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, death @ t=2]` | difference in cumulative incidence of death between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 2 | `cleverly` | -0.000415 to 0.0013 | 0.9544 | 1.0251 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, death @ t=2]` | difference in cumulative incidence of death between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 2 | R `survtmle` | -0.000421 to 0.0013 | 0.9531 | 1.0251 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, death @ t=4]` | difference in cumulative incidence of death between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 4 | `cleverly` | -0.000787 to 0.0016 | 0.9519 | 0.9928 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, death @ t=4]` | difference in cumulative incidence of death between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 4 | R `survtmle` | -0.000808 to 0.0015 | 0.9513 | 0.9932 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 2 | `cleverly` | -0.000912 to 0.0010 | 0.9519 | 1.0157 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 2 | R `survtmle` | -0.000982 to 0.000946 | 0.9475 | 1.0148 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, relapse @ t=4]` | difference in cumulative incidence of relapse between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 4 | `cleverly` | -0.0013 to 0.0014 | 0.9475 | 0.9847 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, relapse @ t=4]` | difference in cumulative incidence of relapse between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 4 | R `survtmle` | -0.0014 to 0.0013 | 0.9475 | 0.9850 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, death @ t=2]` | cumulative incidence of death under the plan assign arm 0 at baseline and hold it at horizon t = 2 | `cleverly` | -0.000980 to 0.000460 | 0.9494 | 1.0075 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, death @ t=2]` | cumulative incidence of death under the plan assign arm 0 at baseline and hold it at horizon t = 2 | R `survtmle` | -0.000976 to 0.000462 | 0.9494 | 1.0083 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, death @ t=4]` | cumulative incidence of death under the plan assign arm 0 at baseline and hold it at horizon t = 4 | `cleverly` | -0.0011 to 0.000869 | 0.9431 | 0.9768 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, death @ t=4]` | cumulative incidence of death under the plan assign arm 0 at baseline and hold it at horizon t = 4 | R `survtmle` | -0.0010 to 0.000877 | 0.9456 | 0.9781 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, relapse @ t=2]` | cumulative incidence of relapse under the plan assign arm 0 at baseline and hold it at horizon t = 2 | `cleverly` | -0.000696 to 0.000665 | 0.9519 | 1.0255 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, relapse @ t=2]` | cumulative incidence of relapse under the plan assign arm 0 at baseline and hold it at horizon t = 2 | R `survtmle` | -0.000647 to 0.000712 | 0.9531 | 1.0275 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, relapse @ t=4]` | cumulative incidence of relapse under the plan assign arm 0 at baseline and hold it at horizon t = 4 | `cleverly` | -0.000883 to 0.000929 | 0.9500 | 1.0063 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, relapse @ t=4]` | cumulative incidence of relapse under the plan assign arm 0 at baseline and hold it at horizon t = 4 | R `survtmle` | -0.000797 to 0.0010 | 0.9513 | 1.0098 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, death @ t=2]` | cumulative incidence of death under the plan assign arm 1 at baseline and hold it at horizon t = 2 | `cleverly` | -0.000304 to 0.000656 | 0.9500 | 1.0170 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, death @ t=2]` | cumulative incidence of death under the plan assign arm 1 at baseline and hold it at horizon t = 2 | R `survtmle` | -0.000307 to 0.000655 | 0.9487 | 1.0145 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, death @ t=4]` | cumulative incidence of death under the plan assign arm 1 at baseline and hold it at horizon t = 4 | `cleverly` | -0.000387 to 0.000968 | 0.9519 | 1.0169 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, death @ t=4]` | cumulative incidence of death under the plan assign arm 1 at baseline and hold it at horizon t = 4 | R `survtmle` | -0.000399 to 0.000958 | 0.9525 | 1.0145 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, relapse @ t=2]` | cumulative incidence of relapse under the plan assign arm 1 at baseline and hold it at horizon t = 2 | `cleverly` | -0.000655 to 0.000727 | 0.9487 | 0.9937 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, relapse @ t=2]` | cumulative incidence of relapse under the plan assign arm 1 at baseline and hold it at horizon t = 2 | R `survtmle` | -0.000677 to 0.000706 | 0.9494 | 0.9919 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, relapse @ t=4]` | cumulative incidence of relapse under the plan assign arm 1 at baseline and hold it at horizon t = 4 | `cleverly` | -0.000883 to 0.0010 | 0.9456 | 0.9692 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, relapse @ t=4]` | cumulative incidence of relapse under the plan assign arm 1 at baseline and hold it at horizon t = 4 | R `survtmle` | -0.000913 to 0.0010 | 0.9463 | 0.9678 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=1]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 1 | `cleverly` | -0.000617 to 0.0012 | 0.9619 | 1.0254 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=1]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 1 | R `survtmle` | -0.000618 to 0.0012 | 0.9619 | 1.0249 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=3]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 3 | `cleverly` | -0.000543 to 0.0023 | 0.9506 | 0.9945 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=3]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 3 | R `survtmle` | -0.000548 to 0.0023 | 0.9500 | 0.9942 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=5]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 5 | `cleverly` | -0.0013 to 0.0019 | 0.9487 | 1.0009 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=5]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 5 | R `survtmle` | -0.0013 to 0.0018 | 0.9494 | 1.0006 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=1]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 1 | `cleverly` | -0.000909 to 0.000506 | 0.9519 | 1.0091 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=1]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 1 | R `survtmle` | -0.000907 to 0.000508 | 0.9525 | 1.0090 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=3]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 3 | `cleverly` | -0.0019 to 0.000165 | 0.9537 | 0.9877 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=3]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 3 | R `survtmle` | -0.0019 to 0.000171 | 0.9525 | 0.9876 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=5]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 5 | `cleverly` | -0.0017 to 0.000440 | 0.9525 | 1.0147 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=5]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 5 | R `survtmle` | -0.0017 to 0.000444 | 0.9537 | 1.0148 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=1]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 1 | `cleverly` | -0.000521 to 0.000688 | 0.9431 | 0.9757 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=1]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 1 | R `survtmle` | -0.000519 to 0.000691 | 0.9431 | 0.9751 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=3]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 3 | `cleverly` | -0.000956 to 0.000924 | 0.9594 | 1.0152 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=3]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 3 | R `survtmle` | -0.000955 to 0.000925 | 0.9600 | 1.0148 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=5]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 5 | `cleverly` | -0.0014 to 0.000779 | 0.9531 | 1.0188 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=5]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 5 | R `survtmle` | -0.0015 to 0.000765 | 0.9544 | 1.0178 | pass |
<!-- /generated -->

All 42 implementation-estimand tests pass, 21 for each implementation. No replication raised an
error, so no cell lost a replicate.

## Agreement with the canonical implementation

<!-- generated: agreement -->
Comparator-backed: a paired comparison with `survtmle` on the same samples. Each row reports estimate equivalence within the declared margin, the RMSE ratio and the coverage difference, and names its comparison kind.

| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | comparison kind | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, death @ t=2]` | difference in cumulative incidence of death between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 2 | 0.000006 | 0.0029 | 1.0011 | 0.0012 | different estimator | 0.000891 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, death @ t=4]` | difference in cumulative incidence of death between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 4 | 0.000021 | 0.0077 | 1.0016 | 0.000625 | different estimator | 0.000912 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 2 | 0.000070 | 0.0311 | 1.0006 | 0.0044 | different estimator | 0.0010 vs 0.0500 | superior |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, relapse @ t=4]` | difference in cumulative incidence of relapse between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 4 | 0.000112 | 0.0362 | 1.0017 | 0 | different estimator | 0.000994 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, death @ t=2]` | cumulative incidence of death under the plan assign arm 0 at baseline and hold it at horizon t = 2 | -0.000003 | 0.0018 | 1.0016 | 0 | different estimator | 0.0021 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, death @ t=4]` | cumulative incidence of death under the plan assign arm 0 at baseline and hold it at horizon t = 4 | -0.000010 | 0.0044 | 1.0021 | -0.0025 | different estimator | 0.000649 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, relapse @ t=2]` | cumulative incidence of relapse under the plan assign arm 0 at baseline and hold it at horizon t = 2 | -0.000048 | 0.0306 | 1.0031 | -0.0012 | different estimator | 0.0043 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, relapse @ t=4]` | cumulative incidence of relapse under the plan assign arm 0 at baseline and hold it at horizon t = 4 | -0.000082 | 0.0391 | 1.0046 | -0.0013 | different estimator | 0.0079 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, death @ t=2]` | cumulative incidence of death under the plan assign arm 1 at baseline and hold it at horizon t = 2 | 0.000003 | 0.0025 | 0.9988 | 0.0012 | different estimator | 0.000842 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, death @ t=4]` | cumulative incidence of death under the plan assign arm 1 at baseline and hold it at horizon t = 4 | 0.000011 | 0.0071 | 0.9992 | -0.000625 | different estimator | 0.000865 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, relapse @ t=2]` | cumulative incidence of relapse under the plan assign arm 1 at baseline and hold it at horizon t = 2 | 0.000021 | 0.0132 | 0.9992 | -0.000625 | different estimator | 0.0039 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, relapse @ t=4]` | cumulative incidence of relapse under the plan assign arm 1 at baseline and hold it at horizon t = 4 | 0.000029 | 0.0130 | 0.9996 | -0.000625 | different estimator | 0.0031 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=1]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 1 | 3.175e-08 | 0.000015 | 1.0002 | 0 | different estimator | 0.000488 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=3]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 3 | 0.000005 | 0.0015 | 1.0002 | 0.000625 | different estimator | 0.000713 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=5]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 5 | 0.000019 | 0.0053 | 1.0006 | -0.000625 | different estimator | 0.000598 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=1]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 1 | -0.000003 | 0.0016 | 1.0007 | -0.000625 | different estimator | 0.000510 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=3]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 3 | -0.000006 | 0.0024 | 1.0003 | 0.0012 | different estimator | 0.000407 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=5]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 5 | -0.000004 | 0.0017 | 1.0004 | -0.0012 | different estimator | 0.000560 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=1]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 1 | -0.000003 | 0.0019 | 1.0003 | 0 | different estimator | 0.0015 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=3]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 3 | -8.924e-07 | 0.000408 | 1.0002 | -0.000625 | different estimator | 0.000261 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=5]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 5 | 0.000015 | 0.0059 | 1.0003 | -0.0012 | different estimator | 0.000821 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `survival_t5__both_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0024 to 0.0013, margin 0.0062, SE ratio 0.9823 | pass |
| `double_robustness` | `survival_t5__both_wrong` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0330 to 0.0366, margin 0.0060, SE ratio 1.0075 | pass |
| `double_robustness` | `survival_t5__mechanism_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0027 to 0.000933, margin 0.0060, SE ratio 1.0206 | pass |
| `double_robustness` | `survival_t5__outcome_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000803 to 0.0028, margin 0.0061, SE ratio 0.9720 | pass |
| `interval_calibration` | `competing_t4__correctly_specified` | positive | difference in the death incidence at visit 4, two causes: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9371 to 0.9541, SE ratio 0.9745 to 1.0264, empirical efficiency ratio 0.9740 to 1.0257, reported efficiency ratio 0.9986 to 1.0004 | pass |
| `interval_calibration` | `survival_t5__correctly_specified` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9445 to 0.9560, SE ratio 0.9859 to 1.0221, empirical efficiency ratio 0.9803 to 1.0162, reported efficiency ratio 1.0015 to 1.0021 | pass |
| `interval_calibration` | `survival_t5__noise_control` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8268 to 0.8464, SE ratio 0.6965 to 0.7223, empirical efficiency ratio 1.3868 to 1.4384, reported efficiency ratio 1.0015 to 1.0021 | pass |
| `interval_calibration` | `survival_t5__shrunken_se_control` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8252 to 0.8448, SE ratio 0.6892 to 0.7157, empirical efficiency ratio 0.9798 to 1.0175, reported efficiency ratio 0.7011 to 0.7015 | pass |
| `power` | `survival_t5__alternative` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `survival_t5__n_1000` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0022, coverage 0.9371 to 0.9747, SE ratio 1.0186 | pass |
| `root_n_and_efficiency` | `survival_t5__n_2000` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000720, coverage 0.9311 to 0.9708, SE ratio 1.0110 | pass |
| `root_n_and_efficiency` | `survival_t5__n_8000` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000264, coverage 0.9252 to 0.9667, SE ratio 0.9897 | pass |
| `root_n_rate` | `survival_t5__empirical_sd` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5286 to -0.4436 | pass |
| `root_n_rate` | `survival_t5__reported_se` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5019 to -0.5005 | pass |
| `simultaneous_coverage` | `all_reported__pointwise_joint_control` | control | every parameter the calibration fit reports: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6522 to 0.6772 | pass |
| `simultaneous_coverage` | `all_reported__simultaneous_band` | positive | every parameter the calibration fit reports: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9392 to 0.9512 | pass |
| `targeting_necessity` | `survival_t5__targeted` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0014 to 0.0025, margin 0.0064 | pass |
| `targeting_necessity` | `survival_t5__untargeted` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.1405 to 0.1405, margin 0 | pass |
<!-- /generated -->

All 18 property cells pass. Each positive cell has a control that fails the same instrument in
the opposite direction.

| family | positive cells | control that must fail |
| --- | --- | --- |
| `interval_calibration` | the visit-5 difference on `survival`, and the death incidence difference at visit 4 on `competing` | reported standard errors scaled by 0.7, and one efficiency-bound unit of added noise |
| `double_robustness` | the hazard correct, the mechanism correct, or both | both wrong |
| `targeting_necessity` | the targeted estimate with a wrong hazard and the correct mechanism | the plug-in of the same fit |
| `root_n_and_efficiency` | n = 2,000 and 8,000 | n = 1,000 |
| `root_n_rate` | the empirical spread and the reported standard error | none. The slope interval must exclude -0.25 |
| `power` | the alternative on `survival` | none |
| `simultaneous_coverage` | the band over the fifteen reported parameters | the pointwise intervals read as a joint set |

The correct mechanism learners are the saturated cell fits. The correct hazard learner is an
additive logistic fit on `survival`, which targeting moves, and the saturated cell fit on
`competing`. A saturated hazard fitted in sample is the maximum-likelihood estimate, so its
initial score is zero and targeting does not move it. The `untargeted` control predicts one hazard
for each node and both arms, so its plug-in contrast is zero in every sample. The necessity
verdict rests on the targeted cell and on its displacement from the plug-in.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications per scenario |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 42 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 42 | of those, passing |
| `paired_tests_total` | 21 | paired comparisons with `survtmle` |
| `paired_tests_passed` | 21 | of those, passing |
| `property_cells_total` | 18 | repeated-sampling property cells |
| `property_cells_passed` | 18 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0543 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9431 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9282 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0773 | highest bootstrap primary SE-ratio endpoint |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:reference_failure_bound` | 0.0100 | declared share of replications a cell of R `survtmle` may lose to a recordable fitting failure |
| `margin:reference_failures_allowed` | 16 | the same bound, in primary replications |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 16 | the same bound, in primary replications |
| `margin:targeting_displacement` | 0.2500 | least targeting must move the estimate |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |

## Limitations

| limit | what it means for use |
| --- | --- |
| per-horizon targeting only | `targeting="one_step"` and the monotone whole-curve estimate are outside this study |
| finite laws with discrete covariates | the evidence covers correct or misspecified low-dimensional logistic and saturated learners. A flexible learner is outside it |
| the paired rows are not an exactness check | `survtmle` fluctuates one cause at a time and stops at `1/n`, so the pairs read under the default margins |
| no cross-fitting, weights or clusters | each is a separate composition and needs its own row |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_hazard/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes, with the `survtmle` runner and image source hashes. Run
`python -m tests.canonical.point_survival_hazard.regenerate` to regenerate the artifacts. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_hazard/replicates.parquet)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_hazard/properties.csv)
carry every published row.
