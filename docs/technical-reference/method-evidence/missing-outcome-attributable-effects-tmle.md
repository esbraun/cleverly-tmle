# Missing-outcome attributable-effect TMLE

This study tests PAR and PAF when outcomes are missing at random. The
[contract](../point-treatment-tmle.md#par-and-paf-with-missing-outcomes) gives the stack and its
conditions. The fit solves the shipped natural-course fluctuation and the shipped arm-mean
fluctuation from one initial fit. It reports both means, PAR and PAF from one stack on the same
rows.

The study has seven scenarios. L1 and L3 are the laws of `tests/studies/mar_arm_indexed_laws.py`.
Each table is indexed by the three levels of $W$ and the arms.

| scenario | law | fit |
| --- | --- | --- |
| `binary_mar_attributable` | L1, two arms | in sample, the law's own nuisances |
| `continuous_mar_attributable` | L1 tables with a Beta outcome | in sample, `q_bounds=(0, 1)`. PAR only, because PAF needs a binary outcome |
| `three_arm_mar_attributable` | L3, three arms | in sample, `reference="low"`, which is not the default reference |
| `binary_mar_attributable_cvtmle` | L1 | stacked CV-TMLE, ten folds, depth-five trees for $Q$, $g$ and $\pi$ |
| `three_arm_mar_attributable_cvtmle` | L3 | stacked CV-TMLE, the same trees |
| `binary_mar_attributable_weighted` | L1 | in sample, fixed weights 0.6, 1.0 and 1.8 by level of $W$ |
| `binary_mar_attributable_clustered` | L1 rows in 100 clusters of 20 that share $W$ | in sample, `id=` |

The stacked scenarios and the weighted scenario refit the samples of the in-sample scenario on the
same law. The in-sample and the stacked fits are therefore paired.

R `tmle` 2.1.1 reports no PAR or PAF. The comparator composes them from two shipped paths. The
population-mean path fits the natural course. The same path with the indicator
$1\{A=a\}\Delta$ fits each arm mean, because its clever covariate is then
$1\{A=a\}\Delta/(g_a\pi_a)$.

The runner forms PAR and PAF from R's own influence curves,
`fit$estimates$IC$IC.EY1`. Both paths use this fit's predictions, so the comparison conditions on
them. The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_attributable/README.md)
gives every argument.

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
<!-- /generated -->

The study declared every budget, seed, law, learner and size before any verdict. The policy is
`gated`. A red cell is diagnosed from the committed rows. A defect in `cleverly` is fixed and the
study is regenerated. Otherwise a declaration switches the study to `reporting` before one re-run
with the same seeds, and the [red-cell ledger](red-cells.md) records the cell.

| family | what it tests |
| --- | --- |
| `mar_robustness` | PAR on L1 with every nuisance correct, with $g$ and $\pi$ wrong, and with $Q$ wrong. Three controls must miss the truth: $Q$ and $g$ wrong, $Q$ and $\pi$ wrong, and `product_only`. In `product_only` the product $g\pi$ is correct at the reference arm and $\pi$ is wrong |
| `root_n_and_efficiency`, `root_n_rate` | PAR at n = 500, 2,000 and 8,000, and the efficiency ratio against the exact EIF SD, 0.67359 |
| `interval_calibration` | PAR and PAF against the calibration bands. A shrunken-SE and a noise control derive from the PAR rows. Two inflated-SE controls drop the cross-covariance of the two parent curves |
| `targeting_necessity` | PAR with a wrong outcome regression, against its untargeted plug-in |
| `missingness_necessity` | PAR with the response declared, against complete-case fits of PAR and PAF |
| `simultaneous_coverage` | the default band of the in-sample L1 fit, the in-sample L3 fit and the stacked L1 fit, each with its pointwise joint control |

The table gives the exact large-sample limits that size each control. `tests/unit/test_mar_attributable_design.py`
recomputes them. One per-replication SD of PAR at n = 2,000 is $0.67359/\sqrt{2000}=0.0151$.

| control | limiting displacement | per-replication SDs |
| --- | ---: | ---: |
| `treatment_wrong` | -0.1354 | 9.0 |
| `observation_wrong` | -0.1968 | 13.1 |
| `product_only` | -0.1243 | 8.3 |
| `par__untargeted` | -0.2280 | 15.1 |
| `par__complete_case_control` | -0.0295 | 1.96 |
| `paf__complete_case_control` | -0.0574 | 1.79 |
| `par__inflated_se_control` | SE ratio 1.951 | not applicable |
| `paf__inflated_se_control` | SE ratio 1.715 | not applicable |

In `treatment_wrong` the natural course stays exact and the reference arm carries the bias. In
`product_only` the reference arm stays exact and the natural course carries it. The band design is
in `tests/unit/test_simultaneous_cell_design.py`. Pointwise joint coverage is 0.8849 on L1 and
0.8148 on L3, and the control power is 1.0 at 2,400 replications. The stacked L1 cell has the
in-sample limit, so it reads the L1 design.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | pending | paired replications |
| `n` | pending | observations per primary replication |
| `independent_tests_passed` | pending | truth tests passing |
| `independent_tests_total` | pending | truth tests reported |
| `paired_tests_passed` | pending | paired comparisons passing |
| `paired_tests_total` | pending | paired comparisons reported |
| `property_cells_passed` | pending | property cells passing |
| `property_cells_total` | pending | property cells reported |
| `max_standardized_bias` | pending | largest primary standardized bias |
| `min_coverage` | pending | lowest primary coverage |
| `max_margin_utilization` | pending | largest paired similarity-margin share |

## Limits

- The study covers finite-support laws with one three-level baseline covariate. The in-sample
  fits use the law's own nuisances, and the stacked fits use trees that can fit the saturated
  model.
- No cross-fitted continuous, clustered or weighted fit is admitted, so none is measured.
- Strata, the bootstrap, DR-TMLE and C-TMLE are refused, so none is measured.
- The product-only rescue of the reference arm is measured as a control, not claimed. The natural
  course needs a correct $\hat\pi$ or a correct $\hat m$.
- The evidence for the scalar `ey_obs` at three arms comes from the three-arm cells. The joint
  `ey_obs` equals the scalar fit bit for bit (`tests/unit/test_attributable_mar_stack.py`).
- No instrument here can detect a violation of missingness at random.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_attributable/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_attributable/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_attributable/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_attributable/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_attributable/properties.csv)
carry the protocol, provenance, and every published row.
