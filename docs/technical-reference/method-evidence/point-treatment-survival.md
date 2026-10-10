# Point-treatment survival

This study validates longitudinal TMLE on a held design. The design has one baseline treatment,
held over every visit, and it reads one follow-up time and one event code per unit
(`TimeToEvent`). The sections
[one baseline treatment held over the nodes](../longitudinal-tmle.md#one-baseline-treatment-held-over-the-nodes)
and [time-to-event input](../longitudinal-tmle.md#time-to-event-input) state the estimator and its
exact-law evidence.

The law is `make_point_survival`. `W1` is Bernoulli(0.5), `W2` is uniform on `0..3`, and the arm is
binary. The discrete-time hazards depend on the visit, the arm and `W`, and dropout at each visit
depends on the arm and `W`. Every truth is an exact finite sum. Every efficiency bound is the
exact standard deviation of the efficient influence function over the law's support.

The study pairs each fit with R `survtmle` 1.1.1, `method = "mean"` (Benkeser, Carone and Gilbert
2018), on the same samples.

## What was compared

| setting | `cleverly` | R `survtmle` |
| --- | --- | --- |
| datasets | 1,600 samples of 2,000 rows for each scenario | the identical rows |
| scenarios | `survival`: five visits, the risks of both arms and their difference at visits 1, 3 and 5. `competing`: four visits and two causes, the incidences and their differences at visits 2 and 4 | the same |
| treatment model | a logistic regression on `W1 + W2` | `glm.trt = "W1 + W2"` |
| censoring model | a logistic regression on the arm, `W1` and `W2` at each visit | one pooled `glm` with one intercept and three slopes for each visit, which factorises by visit into the same model |
| sequential regressions | a quasibinomial regression on `W1 + W2` for each arm. The hazard is logistic in the visit, the arm and `W`, so the iterated means are misspecified on purpose | `glm.ftime = "trt * (W1 + W2)"` |
| targeting | the intercept of each node, with the clever covariate in the loss weight | one logistic fluctuation for each cause, with the clever covariates of both arms |
| intervals | pointwise 95% Wald from the influence curve | the same, from the `survtmle` influence curve |

The two targeting submodels differ. The arms have disjoint support, so the scores separate by
arm and the two solve the same equations along different paths. The paired rows therefore read
under the default margins.

The `initial_estimate` column is not one quantity on the two sides. Here it is the mean of the
node-1 regression, which is fitted on pseudo-outcomes that the later nodes already targeted. The
`survtmle` value is a second call with `Gcomp = TRUE`, which is the fully untargeted recursion. No
verdict reads the column.

Four replications of `competing` (189, 1,008, 1,302 and 1,398) have one sequential regression with
quasi-complete separation. One `W1` cell has no event of one cause at one node, so the coefficient
diverges while the deviance settles. The quasibinomial learner keeps the last iterate with a
warning, as R's `glm` does. The first declared run stopped on replication 189 and published
nothing, and the study re-ran under a fresh declaration.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, death @ t=2]` | difference in cumulative incidence of death between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 2 | `cleverly` | -0.000917 to 0.000859 | 0.9387 | 0.9816 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, death @ t=2]` | difference in cumulative incidence of death between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 2 | R `survtmle`, `method="mean"` | -0.000917 to 0.000859 | 0.9381 | 0.9783 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, death @ t=4]` | difference in cumulative incidence of death between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 4 | `cleverly` | -0.000931 to 0.0014 | 0.9463 | 0.9951 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, death @ t=4]` | difference in cumulative incidence of death between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 4 | R `survtmle`, `method="mean"` | -0.000930 to 0.0014 | 0.9406 | 0.9837 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 2 | `cleverly` | -0.0012 to 0.000758 | 0.9481 | 0.9765 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 2 | R `survtmle`, `method="mean"` | -0.0012 to 0.000759 | 0.9469 | 0.9710 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, relapse @ t=4]` | difference in cumulative incidence of relapse between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 4 | `cleverly` | -0.0023 to 0.000375 | 0.9463 | 0.9869 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, relapse @ t=4]` | difference in cumulative incidence of relapse between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 4 | R `survtmle`, `method="mean"` | -0.0023 to 0.000373 | 0.9375 | 0.9696 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, death @ t=2]` | cumulative incidence of death under the plan assign arm 0 at baseline and hold it at horizon t = 2 | `cleverly` | -0.000915 to 0.000567 | 0.9394 | 0.9781 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, death @ t=2]` | cumulative incidence of death under the plan assign arm 0 at baseline and hold it at horizon t = 2 | R `survtmle`, `method="mean"` | -0.000913 to 0.000568 | 0.9400 | 0.9781 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, death @ t=4]` | cumulative incidence of death under the plan assign arm 0 at baseline and hold it at horizon t = 4 | `cleverly` | -0.0017 to 0.000157 | 0.9444 | 0.9888 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, death @ t=4]` | cumulative incidence of death under the plan assign arm 0 at baseline and hold it at horizon t = 4 | R `survtmle`, `method="mean"` | -0.0017 to 0.000158 | 0.9444 | 0.9889 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, relapse @ t=2]` | cumulative incidence of relapse under the plan assign arm 0 at baseline and hold it at horizon t = 2 | `cleverly` | -0.000655 to 0.000749 | 0.9481 | 0.9940 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, relapse @ t=2]` | cumulative incidence of relapse under the plan assign arm 0 at baseline and hold it at horizon t = 2 | R `survtmle`, `method="mean"` | -0.000654 to 0.000750 | 0.9487 | 0.9939 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, relapse @ t=4]` | cumulative incidence of relapse under the plan assign arm 0 at baseline and hold it at horizon t = 4 | `cleverly` | -0.000362 to 0.0015 | 0.9487 | 0.9831 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, relapse @ t=4]` | cumulative incidence of relapse under the plan assign arm 0 at baseline and hold it at horizon t = 4 | R `survtmle`, `method="mean"` | -0.000360 to 0.0015 | 0.9481 | 0.9832 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, death @ t=2]` | cumulative incidence of death under the plan assign arm 1 at baseline and hold it at horizon t = 2 | `cleverly` | -0.000686 to 0.000279 | 0.9444 | 1.0083 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, death @ t=2]` | cumulative incidence of death under the plan assign arm 1 at baseline and hold it at horizon t = 2 | R `survtmle`, `method="mean"` | -0.000684 to 0.000281 | 0.9387 | 0.9969 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, death @ t=4]` | cumulative incidence of death under the plan assign arm 1 at baseline and hold it at horizon t = 4 | `cleverly` | -0.0012 to 0.000112 | 0.9525 | 1.0267 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, death @ t=4]` | cumulative incidence of death under the plan assign arm 1 at baseline and hold it at horizon t = 4 | R `survtmle`, `method="mean"` | -0.0012 to 0.000114 | 0.9456 | 0.9924 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, relapse @ t=2]` | cumulative incidence of relapse under the plan assign arm 1 at baseline and hold it at horizon t = 2 | `cleverly` | -0.000883 to 0.000491 | 0.9431 | 0.9975 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, relapse @ t=2]` | cumulative incidence of relapse under the plan assign arm 1 at baseline and hold it at horizon t = 2 | R `survtmle`, `method="mean"` | -0.000882 to 0.000492 | 0.9419 | 0.9863 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, relapse @ t=4]` | cumulative incidence of relapse under the plan assign arm 1 at baseline and hold it at horizon t = 4 | `cleverly` | -0.0013 to 0.000554 | 0.9537 | 0.9992 | pass |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, relapse @ t=4]` | cumulative incidence of relapse under the plan assign arm 1 at baseline and hold it at horizon t = 4 | R `survtmle`, `method="mean"` | -0.0013 to 0.000554 | 0.9450 | 0.9649 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=1]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 1 | `cleverly` | -0.0010 to 0.000776 | 0.9619 | 1.0251 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=1]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 1 | R `survtmle`, `method="mean"` | -0.0010 to 0.000776 | 0.9619 | 1.0248 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=3]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 3 | `cleverly` | -0.0015 to 0.0013 | 0.9463 | 0.9948 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=3]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 3 | R `survtmle`, `method="mean"` | -0.0015 to 0.0013 | 0.9450 | 0.9837 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=5]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 5 | `cleverly` | -0.0026 to 0.000558 | 0.9450 | 0.9824 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=5]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 5 | R `survtmle`, `method="mean"` | -0.0026 to 0.000559 | 0.9381 | 0.9568 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=1]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 1 | `cleverly` | -0.000572 to 0.000792 | 0.9637 | 1.0474 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=1]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 1 | R `survtmle`, `method="mean"` | -0.000572 to 0.000792 | 0.9637 | 1.0471 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=3]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 3 | `cleverly` | -0.0010 to 0.0011 | 0.9569 | 0.9999 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=3]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 3 | R `survtmle`, `method="mean"` | -0.0010 to 0.0010 | 0.9569 | 1.0000 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=5]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 5 | `cleverly` | -0.000624 to 0.0015 | 0.9506 | 1.0061 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=5]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 5 | R `survtmle`, `method="mean"` | -0.000628 to 0.0015 | 0.9506 | 1.0065 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=1]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 1 | `cleverly` | -0.000622 to 0.000589 | 0.9406 | 0.9728 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=1]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 1 | R `survtmle`, `method="mean"` | -0.000622 to 0.000588 | 0.9406 | 0.9725 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=3]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 3 | `cleverly` | -0.0011 to 0.000850 | 0.9475 | 0.9878 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=3]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 3 | R `survtmle`, `method="mean"` | -0.0011 to 0.000850 | 0.9406 | 0.9639 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=5]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 5 | `cleverly` | -0.0017 to 0.000572 | 0.9469 | 0.9879 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=5]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 5 | R `survtmle`, `method="mean"` | -0.0017 to 0.000569 | 0.9287 | 0.9378 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, death @ t=2]` | difference in cumulative incidence of death between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 2 | -1.396e-07 | 0.000068 | 1.0005 | 0.000625 | 0.0069 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, death @ t=4]` | difference in cumulative incidence of death between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 4 | -6.531e-07 | 0.000241 | 1.0005 | 0.0056 | 0.0233 vs 0.0500 | superior |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 2 | -3.299e-07 | 0.000142 | 1.0003 | 0.0012 | 0.0113 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `ate_regimen[arm1 vs arm0, relapse @ t=4]` | difference in cumulative incidence of relapse between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 4 | 0.000001 | 0.000395 | 1.0004 | 0.0088 | 0.0353 vs 0.0500 | superior |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, death @ t=2]` | cumulative incidence of death under the plan assign arm 0 at baseline and hold it at horizon t = 2 | -0.000002 | 0.000896 | 1.0004 | -0.000625 | 0.000153 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, death @ t=4]` | cumulative incidence of death under the plan assign arm 0 at baseline and hold it at horizon t = 4 | -0.000001 | 0.000496 | 1.0005 | 0 | 0.000195 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, relapse @ t=2]` | cumulative incidence of relapse under the plan assign arm 0 at baseline and hold it at horizon t = 2 | -7.538e-07 | 0.000462 | 1.0003 | -0.000625 | 0.000291 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm0, relapse @ t=4]` | cumulative incidence of relapse under the plan assign arm 0 at baseline and hold it at horizon t = 4 | -0.000002 | 0.000877 | 1.0004 | 0.000625 | 0.000205 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, death @ t=2]` | cumulative incidence of death under the plan assign arm 1 at baseline and hold it at horizon t = 2 | -0.000002 | 0.0015 | 1.0003 | 0.0056 | 0.0069 vs 0.0500 | superior |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, death @ t=4]` | cumulative incidence of death under the plan assign arm 1 at baseline and hold it at horizon t = 4 | -0.000002 | 0.0011 | 1.0004 | 0.0069 | 0.0168 vs 0.0500 | superior |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, relapse @ t=2]` | cumulative incidence of relapse under the plan assign arm 1 at baseline and hold it at horizon t = 2 | -0.000001 | 0.000678 | 1.0004 | 0.0012 | 0.0232 vs 0.0500 | equivalent |
| four visits, two competing causes, a held binary treatment, visit dropout | `cif_regimen[arm1, relapse @ t=4]` | cumulative incidence of relapse under the plan assign arm 1 at baseline and hold it at horizon t = 4 | -6.768e-07 | 0.000311 | 1.0004 | 0.0088 | 0.0702 vs 0.0500 **>** | superior |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=1]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 1 | 4.393e-08 | 0.000021 | 1.0000 | 0 | 0.000019 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=3]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 3 | 2.294e-07 | 0.000070 | 1.0004 | 0.0013 | 0.0227 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=5]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 5 | -0.000001 | 0.000384 | 1.0005 | 0.0069 | 0.0471 vs 0.0500 | superior |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=1]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 1 | 1.189e-09 | 7.490e-07 | 1.0000 | 0 | 0.000023 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=3]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 3 | 4.474e-07 | 0.000186 | 1.0004 | 0 | 0.000215 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=5]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 5 | 0.000004 | 0.0016 | 1.0006 | 0 | 0.000894 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=1]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 1 | 4.512e-08 | 0.000032 | 1.0000 | 0 | 0.000510 vs 0.0500 | equivalent |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=3]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 3 | 6.768e-07 | 0.000301 | 1.0004 | 0.0069 | 0.0490 vs 0.0500 | superior |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=5]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 5 | 0.000003 | 0.000982 | 1.0003 | 0.0181 | 0.0579 vs 0.0500 **>** | superior |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `clustered_inference` | `clustered_t5__cluster_robust` | positive | arm-0 risk at visit 5, 25 clusters of 80 that share W: five-fold point-treatment TMLE with cluster-robust ATE inference | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9385 to 0.9662, SE ratio 0.9560 to 1.0478, paired coverage gain 0.0838 to 0.1237 | pass |
| `clustered_inference` | `clustered_t5__iid_control` | control | arm-0 risk at visit 5, 25 clusters of 80 that share W: the identical rows, point estimates, and influence curves treated as independent | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.8263 to 0.8728, SE ratio 0.7084 to 0.7747, paired coverage gain 0.0838 to 0.1237 | pass |
| `double_robustness` | `survival_t5__both_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0011 to 0.0026, margin 0.0062, SE ratio 0.9769 | pass |
| `double_robustness` | `survival_t5__both_wrong` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0316 to 0.0352, margin 0.0061, SE ratio 0.9909 | pass |
| `double_robustness` | `survival_t5__mechanism_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0016 to 0.0020, margin 0.0061, SE ratio 1.0083 | pass |
| `double_robustness` | `survival_t5__outcome_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0024 to 0.0012, margin 0.0062, SE ratio 0.9626 | pass |
| `interval_calibration` | `competing_t4__correctly_specified` | positive | difference in the death incidence at visit 4, two causes: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9421 to 0.9689, SE ratio 0.9662 to 1.0596, empirical efficiency ratio 0.9429 to 1.0341, reported efficiency ratio 0.9976 to 1.0008 | pass |
| `interval_calibration` | `competing_t4__noise_control` | control | difference in the death incidence at visit 4, two causes: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8039 to 0.8529, SE ratio 0.6808 to 0.7449, empirical efficiency ratio 1.3415 to 1.4677, reported efficiency ratio 0.9976 to 1.0008 | pass |
| `interval_calibration` | `competing_t4__shrunken_se_control` | control | difference in the death incidence at visit 4, two causes: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8078 to 0.8564, SE ratio 0.6775 to 0.7423, empirical efficiency ratio 0.9421 to 1.0325, reported efficiency ratio 0.6983 to 0.7005 | pass |
| `interval_calibration` | `continuous_t4__correctly_specified` | positive | difference at grid time 3, exponential event time, visit dropout: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9293 to 0.9592, SE ratio 0.9411 to 1.0294, empirical efficiency ratio 0.9708 to 1.0618, reported efficiency ratio 0.9986 to 1.0000 | pass |
| `interval_calibration` | `continuous_t4__noise_control` | control | difference at grid time 3, exponential event time, visit dropout: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8012 to 0.8505, SE ratio 0.6567 to 0.7199, empirical efficiency ratio 1.3880 to 1.5217, reported efficiency ratio 0.9986 to 1.0000 | pass |
| `interval_calibration` | `continuous_t4__shrunken_se_control` | control | difference at grid time 3, exponential event time, visit dropout: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7888 to 0.8393, SE ratio 0.6588 to 0.7212, empirical efficiency ratio 0.9702 to 1.0616, reported efficiency ratio 0.6990 to 0.7000 | pass |
| `interval_calibration` | `end_of_study__correctly_specified` | positive | held end-of-study difference after two censoring nodes: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9307 to 0.9603, SE ratio 0.9306 to 1.0218, empirical efficiency ratio 0.9786 to 1.0746, reported efficiency ratio 0.9993 to 1.0005 | pass |
| `interval_calibration` | `end_of_study__noise_control` | control | held end-of-study difference after two censoring nodes: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8111 to 0.8594, SE ratio 0.6739 to 0.7385, empirical efficiency ratio 1.3542 to 1.4837, reported efficiency ratio 0.9993 to 1.0005 | pass |
| `interval_calibration` | `end_of_study__shrunken_se_control` | control | held end-of-study difference after two censoring nodes: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7914 to 0.8417, SE ratio 0.6516 to 0.7139, empirical efficiency ratio 0.9805 to 1.0742, reported efficiency ratio 0.6995 to 0.7004 | pass |
| `interval_calibration` | `rmst_5__correctly_specified` | positive | RMST difference up to visit 5, five-visit held law: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9429 to 0.9546, SE ratio 0.9842 to 1.0206, empirical efficiency ratio 0.9795 to 1.0156, reported efficiency ratio 0.9993 to 1.0001 | pass |
| `interval_calibration` | `rmst_5__noise_control` | control | RMST difference up to visit 5, five-visit held law: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8262 to 0.8458, SE ratio 0.6983 to 0.7248, empirical efficiency ratio 1.3793 to 1.4316, reported efficiency ratio 0.9993 to 1.0002 | pass |
| `interval_calibration` | `rmst_5__shrunken_se_control` | control | RMST difference up to visit 5, five-visit held law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8226 to 0.8423, SE ratio 0.6885 to 0.7144, empirical efficiency ratio 0.9794 to 1.0164, reported efficiency ratio 0.6995 to 0.7001 | pass |
| `interval_calibration` | `survival_t5__correctly_specified` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9416 to 0.9534, SE ratio 0.9774 to 1.0144, empirical efficiency ratio 0.9857 to 1.0229, reported efficiency ratio 0.9996 to 1.0002 | pass |
| `interval_calibration` | `survival_t5__noise_control` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8181 to 0.8380, SE ratio 0.6835 to 0.7098, empirical efficiency ratio 1.4086 to 1.4628, reported efficiency ratio 0.9996 to 1.0002 | pass |
| `interval_calibration` | `survival_t5__shrunken_se_control` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8196 to 0.8395, SE ratio 0.6840 to 0.7097, empirical efficiency ratio 0.9862 to 1.0231, reported efficiency ratio 0.6997 to 0.7001 | pass |
| `interval_calibration` | `three_arm_t3__correctly_specified` | positive | difference of arm 2 and arm 0 at visit 3, three arms with L_t: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9314 to 0.9608, SE ratio 0.9513 to 1.0400, empirical efficiency ratio 0.9610 to 1.0509, reported efficiency ratio 0.9986 to 1.0004 | pass |
| `interval_calibration` | `three_arm_t3__noise_control` | control | difference of arm 2 and arm 0 at visit 3, three arms with L_t: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8045 to 0.8535, SE ratio 0.6770 to 0.7410, empirical efficiency ratio 1.3486 to 1.4764, reported efficiency ratio 0.9986 to 1.0004 | pass |
| `interval_calibration` | `three_arm_t3__shrunken_se_control` | control | difference of arm 2 and arm 0 at visit 3, three arms with L_t: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7888 to 0.8393, SE ratio 0.6661 to 0.7270, empirical efficiency ratio 0.9627 to 1.0506, reported efficiency ratio 0.6990 to 0.7003 | pass |
| `interval_calibration` | `weighted_t5__correctly_specified` | positive | difference at visit 5 under the weight 1 + W1 / 2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9421 to 0.9689, SE ratio 0.9807 to 1.0762, empirical efficiency ratio 0.9300 to 1.0204, reported efficiency ratio 1.0000 to 1.0015 | **fail** |
| `interval_calibration` | `weighted_t5__noise_control` | control | difference at visit 5 under the weight 1 + W1 / 2: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8085 to 0.8570, SE ratio 0.6797 to 0.7429, empirical efficiency ratio 1.3471 to 1.4722, reported efficiency ratio 0.9999 to 1.0014 | pass |
| `interval_calibration` | `weighted_t5__shrunken_se_control` | control | difference at visit 5 under the weight 1 + W1 / 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8197 to 0.8670, SE ratio 0.6875 to 0.7533, empirical efficiency ratio 0.9299 to 1.0187, reported efficiency ratio 0.7000 to 0.7010 | pass |
| `power` | `survival_t5__alternative` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `survival_t5__n_1000` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0010, coverage 0.9462 to 0.9805, SE ratio 1.0472 | pass |
| `root_n_and_efficiency` | `survival_t5__n_2000` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000173, coverage 0.9311 to 0.9708, SE ratio 0.9952 | pass |
| `root_n_and_efficiency` | `survival_t5__n_8000` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000019, coverage 0.9401 to 0.9767, SE ratio 1.0043 | pass |
| `root_n_rate` | `survival_t5__empirical_sd` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5236 to -0.4440 | pass |
| `root_n_rate` | `survival_t5__reported_se` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5007 to -0.4993 | pass |
| `simultaneous_coverage` | `all_reported__pointwise_joint_control` | control | every parameter the calibration fit reports: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6526 to 0.6775 | pass |
| `simultaneous_coverage` | `all_reported__simultaneous_band` | positive | every parameter the calibration fit reports: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9400 to 0.9520 | pass |
<!-- /generated -->

The positive cells fit the saturated cell means, which are correct on these finite laws. Each
calibration label reads one estimand on one law.

| label | estimand |
| --- | --- |
| `survival_t5` | the difference of the two arm risks at visit 5 |
| `rmst_5` | the difference of the restricted mean survival times at visit 5, on the `survival_t5` fits |
| `competing_t4` | the difference of the death incidences at visit 4 |
| `three_arm_t3` | arm 2 against arm 0 at visit 3, with a binary covariate measured at each visit, on the wide layout |
| `continuous_t4` | the difference at grid time 3, with an exponential event time binned onto the grid |
| `end_of_study` | the difference of the held end-of-study means after two censoring nodes |
| `weighted_t5` | the visit-5 difference on the law tilted by the weight `1 + W1 / 2` |
| `clustered_t5` | the arm-0 risk at visit 5 on 25 clusters of 80 rows that share `W` |

The `weighted_t5__correctly_specified` calibration cell is red on its SE-ratio band. The SE ratio
is 1.027, and its 99% interval is 0.981 to 1.076 against an upper bound of 1.07. The coverage is
0.957. The reported standard error is the weighted efficiency bound to 0.07% (ratio 1.0007). The
empirical spread is 2.5% below the bound, and its interval, 0.930 to 1.020, contains the bound.

The diagnostic
[`tests/diagnostics/x13_weighted_calibration/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/x13_weighted_calibration)
refits the cell on 1,000 fresh draws, and its module records the seeds and the command. It reads
an SE ratio of 1.022, inside the band, an empirical efficiency ratio of 0.979, a reported
efficiency ratio of 1.000 and a coverage of 0.957. The diagnosis finds no defect: at 1,600
replications the interval reaches past the band from a point of 1.027. By the declared route the
cell stays red under `reporting`, with the owner row `X13-finite-sample`.

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
| `property_cells_total` | 35 | repeated-sampling property cells |
| `property_cells_passed` | 34 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0538 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9287 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.8978 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0992 | highest bootstrap primary SE-ratio endpoint |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
| `margin:clustered_coverage_gain` | 0.0300 | paired coverage-gain floor |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:iid_control_se_ceiling` | 0.8000 | IID-control SE-ratio ceiling |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |

## Limitations

| limit | what it means for use |
| --- | --- |
| one baseline treatment on finite laws | the evidence covers binary and three-arm baseline treatments on discrete covariates. A continuous dose has fast-tier evidence only |
| the paired rows are not an exactness check | the targeting submodels differ by construction, so the pairs read under the default margins |
| per-horizon targeting | each horizon has its own backward pass, so the curve is not constrained to be monotone. [X14](../../roadmap.md#x14-hazard-based-and-monotone-survival-curves) owns hazard-based and whole-curve targeting |
| discrete time | a continuous event time is binned onto a declared grid. The `continuous_t4` cell measures the binned parameter, not a continuous-time one |
| one calibration cell is red under `reporting` | the `weighted_t5` SE-ratio interval reaches past its band. The reported standard error equals the efficiency bound. The `X13-finite-sample` owner holds it |
| `survtmle`'s `bounds=` is not paired | [X22](../../roadmap.md#x22-targeting-and-fitting-options) part (h) owns it |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. Run `python -m tests.canonical.point_survival.regenerate` to regenerate the artifacts. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival/replicates.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival/properties.csv)
carry every published row.
