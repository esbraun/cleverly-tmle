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
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1: two-arm binary-outcome law with MAR outcomes, in sample | `ey0` | counterfactual mean under no treatment | `cleverly` missing-outcome attributable-effect TMLE | -0.0017 to 0.0028 | 0.9350 | 0.9652 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, in sample | `ey0` | counterfactual mean under no treatment | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0017 to 0.0028 | 0.9350 | 0.9652 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, in sample | `ey_obs` | observed outcome mean under the natural course | `cleverly` missing-outcome attributable-effect TMLE | -0.0013 to 0.0019 | 0.9450 | 0.9708 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, in sample | `ey_obs` | observed outcome mean under the natural course | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0013 to 0.0019 | 0.9450 | 0.9708 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, in sample | `paf` | population attributable fraction | `cleverly` missing-outcome attributable-effect TMLE | -0.0031 to 0.0029 | 0.9375 | 0.9872 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, in sample | `paf` | population attributable fraction | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0031 to 0.0029 | 0.9375 | 0.9872 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, in sample | `par` | population attributable risk | `cleverly` missing-outcome attributable-effect TMLE | -0.0016 to 0.0012 | 0.9500 | 1.0034 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, in sample | `par` | population attributable risk | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0016 to 0.0012 | 0.9500 | 1.0034 | pass |
| L1 rows in 100 clusters of 20 that share W, in sample with id= | `ey0` | counterfactual mean under no treatment | `cleverly` missing-outcome attributable-effect TMLE | -0.0040 to 0.0010 | 0.9513 | 0.9808 | pass |
| L1 rows in 100 clusters of 20 that share W, in sample with id= | `ey0` | counterfactual mean under no treatment | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0040 to 0.0010 | 0.9513 | 0.9808 | pass |
| L1 rows in 100 clusters of 20 that share W, in sample with id= | `ey_obs` | observed outcome mean under the natural course | `cleverly` missing-outcome attributable-effect TMLE | -0.0028 to 0.000647 | 0.9425 | 1.0006 | pass |
| L1 rows in 100 clusters of 20 that share W, in sample with id= | `ey_obs` | observed outcome mean under the natural course | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0028 to 0.000647 | 0.9425 | 1.0006 | pass |
| L1 rows in 100 clusters of 20 that share W, in sample with id= | `paf` | population attributable fraction | `cleverly` missing-outcome attributable-effect TMLE | -0.0018 to 0.0055 | 0.9437 | 0.9705 | pass |
| L1 rows in 100 clusters of 20 that share W, in sample with id= | `paf` | population attributable fraction | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0018 to 0.0055 | 0.9437 | 0.9705 | pass |
| L1 rows in 100 clusters of 20 that share W, in sample with id= | `par` | population attributable risk | `cleverly` missing-outcome attributable-effect TMLE | -0.0013 to 0.0022 | 0.9487 | 0.9732 | pass |
| L1 rows in 100 clusters of 20 that share W, in sample with id= | `par` | population attributable risk | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0013 to 0.0022 | 0.9487 | 0.9732 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees | `ey0` | counterfactual mean under no treatment | `cleverly` missing-outcome attributable-effect TMLE | -0.0017 to 0.0028 | 0.9413 | 0.9742 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees | `ey0` | counterfactual mean under no treatment | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0017 to 0.0028 | 0.9413 | 0.9742 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees | `ey_obs` | observed outcome mean under the natural course | `cleverly` missing-outcome attributable-effect TMLE | -0.0013 to 0.0019 | 0.9475 | 0.9764 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees | `ey_obs` | observed outcome mean under the natural course | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0013 to 0.0019 | 0.9475 | 0.9764 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees | `paf` | population attributable fraction | `cleverly` missing-outcome attributable-effect TMLE | -0.0032 to 0.0027 | 0.9463 | 0.9977 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees | `paf` | population attributable fraction | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0032 to 0.0027 | 0.9463 | 0.9977 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees | `par` | population attributable risk | `cleverly` missing-outcome attributable-effect TMLE | -0.0016 to 0.0011 | 0.9525 | 1.0133 | pass |
| L1: two-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees | `par` | population attributable risk | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0016 to 0.0011 | 0.9525 | 1.0133 | pass |
| L1 with fixed analysis weights 0.6, 1.0 and 1.8 by level of W, in sample | `ey0` | counterfactual mean under no treatment | `cleverly` missing-outcome attributable-effect TMLE | -0.0016 to 0.0024 | 0.9363 | 0.9491 | pass |
| L1 with fixed analysis weights 0.6, 1.0 and 1.8 by level of W, in sample | `ey0` | counterfactual mean under no treatment | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0016 to 0.0024 | 0.9363 | 0.9491 | pass |
| L1 with fixed analysis weights 0.6, 1.0 and 1.8 by level of W, in sample | `ey_obs` | observed outcome mean under the natural course | `cleverly` missing-outcome attributable-effect TMLE | -0.0013 to 0.0017 | 0.9463 | 0.9594 | pass |
| L1 with fixed analysis weights 0.6, 1.0 and 1.8 by level of W, in sample | `ey_obs` | observed outcome mean under the natural course | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0013 to 0.0017 | 0.9463 | 0.9594 | pass |
| L1 with fixed analysis weights 0.6, 1.0 and 1.8 by level of W, in sample | `paf` | population attributable fraction | `cleverly` missing-outcome attributable-effect TMLE | -0.0029 to 0.0024 | 0.9450 | 0.9847 | pass |
| L1 with fixed analysis weights 0.6, 1.0 and 1.8 by level of W, in sample | `paf` | population attributable fraction | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0029 to 0.0024 | 0.9450 | 0.9847 | pass |
| L1 with fixed analysis weights 0.6, 1.0 and 1.8 by level of W, in sample | `par` | population attributable risk | `cleverly` missing-outcome attributable-effect TMLE | -0.0015 to 0.0011 | 0.9500 | 0.9959 | pass |
| L1 with fixed analysis weights 0.6, 1.0 and 1.8 by level of W, in sample | `par` | population attributable risk | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0015 to 0.0011 | 0.9500 | 0.9959 | pass |
| L1 tables with a bounded continuous Beta outcome and MAR outcomes, in sample | `ey0` | counterfactual mean under no treatment | `cleverly` missing-outcome attributable-effect TMLE | -0.000361 to 0.000640 | 0.9600 | 1.0285 | pass |
| L1 tables with a bounded continuous Beta outcome and MAR outcomes, in sample | `ey0` | counterfactual mean under no treatment | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.000361 to 0.000640 | 0.9600 | 1.0285 | pass |
| L1 tables with a bounded continuous Beta outcome and MAR outcomes, in sample | `ey_obs` | observed outcome mean under the natural course | `cleverly` missing-outcome attributable-effect TMLE | -0.000330 to 0.000688 | 0.9513 | 0.9826 | pass |
| L1 tables with a bounded continuous Beta outcome and MAR outcomes, in sample | `ey_obs` | observed outcome mean under the natural course | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.000330 to 0.000688 | 0.9513 | 0.9826 | pass |
| L1 tables with a bounded continuous Beta outcome and MAR outcomes, in sample | `par` | population attributable risk | `cleverly` missing-outcome attributable-effect TMLE | -0.000483 to 0.000564 | 0.9413 | 0.9512 | pass |
| L1 tables with a bounded continuous Beta outcome and MAR outcomes, in sample | `par` | population attributable risk | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.000483 to 0.000564 | 0.9413 | 0.9512 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` missing-outcome attributable-effect TMLE | -0.0028 to 0.0022 | 0.9413 | 0.9716 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `ey[high]` | counterfactual mean under treatment arm 'high' | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0028 to 0.0022 | 0.9413 | 0.9716 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` missing-outcome attributable-effect TMLE | -0.000996 to 0.0038 | 0.9563 | 1.0303 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `ey[low]` | counterfactual mean under treatment arm 'low' | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.000996 to 0.0038 | 0.9563 | 1.0303 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `ey[mid]` | counterfactual mean under treatment arm 'mid' | `cleverly` missing-outcome attributable-effect TMLE | -0.0016 to 0.0027 | 0.9437 | 1.0117 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `ey[mid]` | counterfactual mean under treatment arm 'mid' | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0016 to 0.0027 | 0.9437 | 1.0117 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `ey_obs` | observed outcome mean under the natural course | `cleverly` missing-outcome attributable-effect TMLE | -0.000801 to 0.0019 | 0.9625 | 1.0622 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `ey_obs` | observed outcome mean under the natural course | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.000801 to 0.0019 | 0.9625 | 1.0622 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `paf[low]` | population attributable fraction, natural course versus reference arm 'low' | `cleverly` missing-outcome attributable-effect TMLE | -0.0052 to 0.0021 | 0.9425 | 1.0036 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `paf[low]` | population attributable fraction, natural course versus reference arm 'low' | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0052 to 0.0021 | 0.9425 | 1.0036 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `par[low]` | population attributable risk, natural course versus reference arm 'low' | `cleverly` missing-outcome attributable-effect TMLE | -0.0027 to 0.001000 | 0.9463 | 1.0002 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `par[low]` | population attributable risk, natural course versus reference arm 'low' | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0027 to 0.001000 | 0.9463 | 1.0002 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` missing-outcome attributable-effect TMLE | -0.0027 to 0.0023 | 0.9437 | 0.9832 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `ey[high]` | counterfactual mean under treatment arm 'high' | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0027 to 0.0023 | 0.9437 | 0.9832 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` missing-outcome attributable-effect TMLE | -0.0011 to 0.0037 | 0.9587 | 1.0387 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `ey[low]` | counterfactual mean under treatment arm 'low' | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0011 to 0.0037 | 0.9587 | 1.0387 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `ey[mid]` | counterfactual mean under treatment arm 'mid' | `cleverly` missing-outcome attributable-effect TMLE | -0.0016 to 0.0027 | 0.9487 | 1.0273 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `ey[mid]` | counterfactual mean under treatment arm 'mid' | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0016 to 0.0027 | 0.9487 | 1.0273 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `ey_obs` | observed outcome mean under the natural course | `cleverly` missing-outcome attributable-effect TMLE | -0.000792 to 0.0019 | 0.9587 | 1.0650 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `ey_obs` | observed outcome mean under the natural course | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.000792 to 0.0019 | 0.9587 | 1.0650 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `paf[low]` | population attributable fraction, natural course versus reference arm 'low' | `cleverly` missing-outcome attributable-effect TMLE | -0.0050 to 0.0024 | 0.9500 | 1.0147 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `paf[low]` | population attributable fraction, natural course versus reference arm 'low' | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0050 to 0.0024 | 0.9500 | 1.0147 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `par[low]` | population attributable risk, natural course versus reference arm 'low' | `cleverly` missing-outcome attributable-effect TMLE | -0.0026 to 0.0012 | 0.9563 | 1.0110 | pass |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `par[low]` | population attributable risk, natural course versus reference arm 'low' | R `tmle` population-mean fits for the natural course and each arm, composed by the delta method from R's own influence curves | -0.0026 to 0.0012 | 0.9563 | 1.0110 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1: two-arm binary-outcome law with MAR outcomes, in sample | `ey0` | counterfactual mean under no treatment | -3.944e-10 | 1.068e-07 | 1.0000 | 0 | 2.374e-08 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome law with MAR outcomes, in sample | `ey_obs` | observed outcome mean under the natural course | -4.449e-13 | 1.668e-10 | 1.0000 | 0 | 1.490e-11 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome law with MAR outcomes, in sample | `paf` | population attributable fraction | 7.822e-10 | 1.602e-07 | 1.0000 | 0 | 3.213e-08 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome law with MAR outcomes, in sample | `par` | population attributable risk | 3.940e-10 | 1.750e-07 | 1.0000 | 0 | 3.921e-09 vs 0.0500 | equivalent |
| L1 rows in 100 clusters of 20 that share W, in sample with id= | `ey0` | counterfactual mean under no treatment | -4.499e-10 | 1.079e-07 | 1.0000 | 0 | 2.021e-08 vs 0.0500 | equivalent |
| L1 rows in 100 clusters of 20 that share W, in sample with id= | `ey_obs` | observed outcome mean under the natural course | -4.917e-13 | 1.747e-10 | 1.0000 | 0 | 1.071e-10 vs 0.0500 | equivalent |
| L1 rows in 100 clusters of 20 that share W, in sample with id= | `paf` | population attributable fraction | 9.013e-10 | 1.498e-07 | 1.0000 | 0 | 2.827e-08 vs 0.0500 | equivalent |
| L1 rows in 100 clusters of 20 that share W, in sample with id= | `par` | population attributable risk | 4.494e-10 | 1.594e-07 | 1.0000 | 0 | 2.638e-08 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees | `ey0` | counterfactual mean under no treatment | -3.170e-10 | 8.565e-08 | 1.0000 | 0 | 6.937e-09 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees | `ey_obs` | observed outcome mean under the natural course | -2.225e-11 | 8.322e-09 | 1.0000 | 0 | 6.049e-10 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees | `paf` | population attributable fraction | 6.048e-10 | 1.235e-07 | 1.0000 | 0 | 1.337e-08 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees | `par` | population attributable risk | 2.948e-10 | 1.306e-07 | 1.0000 | 0 | 3.345e-09 vs 0.0500 | equivalent |
| L1 with fixed analysis weights 0.6, 1.0 and 1.8 by level of W, in sample | `ey0` | counterfactual mean under no treatment | -1.203e-10 | 3.667e-08 | 1.0000 | 0 | 3.785e-09 vs 0.0500 | equivalent |
| L1 with fixed analysis weights 0.6, 1.0 and 1.8 by level of W, in sample | `ey_obs` | observed outcome mean under the natural course | -1.277e-12 | 5.198e-10 | 1.0000 | 0 | 5.305e-11 vs 0.0500 | equivalent |
| L1 with fixed analysis weights 0.6, 1.0 and 1.8 by level of W, in sample | `paf` | population attributable fraction | 2.556e-10 | 5.802e-08 | 1.0000 | 0 | 5.482e-09 vs 0.0500 | equivalent |
| L1 with fixed analysis weights 0.6, 1.0 and 1.8 by level of W, in sample | `par` | population attributable risk | 1.190e-10 | 5.743e-08 | 1.0000 | 0 | 4.419e-09 vs 0.0500 | equivalent |
| L1 tables with a bounded continuous Beta outcome and MAR outcomes, in sample | `ey0` | counterfactual mean under no treatment | -9.118e-14 | 1.109e-10 | 1.0000 | 0 | 1.819e-10 vs 0.0500 | equivalent |
| L1 tables with a bounded continuous Beta outcome and MAR outcomes, in sample | `ey_obs` | observed outcome mean under the natural course | -4.240e-14 | 5.069e-11 | 1.0000 | 0 | 1.520e-11 vs 0.0500 | equivalent |
| L1 tables with a bounded continuous Beta outcome and MAR outcomes, in sample | `par` | population attributable risk | 4.878e-14 | 5.670e-11 | 1.0000 | 0 | 4.609e-11 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `ey[high]` | counterfactual mean under treatment arm 'high' | 1.108e-09 | 2.718e-07 | 1.0000 | 0 | 2.044e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `ey[low]` | counterfactual mean under treatment arm 'low' | -3.087e-10 | 7.885e-08 | 1.0000 | 0 | 1.509e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `ey[mid]` | counterfactual mean under treatment arm 'mid' | 1.455e-11 | 4.095e-09 | 1.0000 | 0 | 1.244e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `ey_obs` | observed outcome mean under the natural course | -3.927e-12 | 1.786e-09 | 1.0000 | 0 | 7.039e-11 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `paf[low]` | population attributable fraction, natural course versus reference arm 'low' | 5.686e-10 | 9.558e-08 | 1.0000 | 0 | 1.801e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low | `par[low]` | population attributable risk, natural course versus reference arm 'low' | 3.048e-10 | 1.003e-07 | 1.0000 | 0 | 1.780e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `ey[high]` | counterfactual mean under treatment arm 'high' | 2.657e-10 | 6.481e-08 | 1.0000 | 0 | 1.290e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `ey[low]` | counterfactual mean under treatment arm 'low' | -1.729e-10 | 4.368e-08 | 1.0000 | 0 | 1.042e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `ey[mid]` | counterfactual mean under treatment arm 'mid' | 3.801e-11 | 1.066e-08 | 1.0000 | 0 | 3.196e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `ey_obs` | observed outcome mean under the natural course | -4.712e-11 | 2.132e-08 | 1.0000 | 0 | 8.612e-10 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `paf[low]` | population attributable fraction, natural course versus reference arm 'low' | 2.610e-10 | 4.340e-08 | 1.0000 | 0 | 1.383e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees, reference arm low | `par[low]` | population attributable risk, natural course versus reference arm 'low' | 1.258e-10 | 4.096e-08 | 1.0000 | 0 | 1.494e-09 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `paf__correctly_specified` | positive | population attributable fraction: all three required nuisance functions are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9369 to 0.9555, SE ratio 0.9635 to 1.0214, empirical efficiency ratio 0.9796 to 1.0386, reported efficiency ratio 0.9989 to 1.0022 | pass |
| `interval_calibration` | `paf__inflated_se_control` | control | population attributable fraction: the cross-covariance of the two parent curves is dropped, so the standard error adds the natural-course and reference-arm variances as if independent | the SE-ratio interval must fall above the calibration band | coverage 0.9977 to 1.0000, SE ratio 1.6503 to 1.7477, empirical efficiency ratio 0.9808 to 1.0383, reported efficiency ratio 1.7111 to 1.7169 | pass |
| `interval_calibration` | `par__correctly_specified` | positive | population attributable risk: all three required nuisance functions are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9359 to 0.9546, SE ratio 0.9665 to 1.0225, empirical efficiency ratio 0.9776 to 1.0343, reported efficiency ratio 0.9986 to 1.0005 | pass |
| `interval_calibration` | `par__inflated_se_control` | control | population attributable risk: the cross-covariance of the two parent curves is dropped, so the standard error adds the natural-course and reference-arm variances as if independent | the SE-ratio interval must fall above the calibration band | coverage 0.9987 to 1, SE ratio 1.8847 to 1.9927, empirical efficiency ratio 0.9769 to 1.0329, reported efficiency ratio 1.9441 to 1.9488 | pass |
| `interval_calibration` | `par__noise_control` | control | population attributable risk: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8103 to 0.8414, SE ratio 0.6808 to 0.7212, empirical efficiency ratio 1.3859 to 1.4684, reported efficiency ratio 0.9986 to 1.0005 | pass |
| `interval_calibration` | `par__shrunken_se_control` | control | population attributable risk: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8088 to 0.8400, SE ratio 0.6775 to 0.7161, empirical efficiency ratio 0.9771 to 1.0327, reported efficiency ratio 0.6990 to 0.7004 | pass |
| `mar_robustness` | `both_correct` | positive | the outcome regression, treatment mechanism and observation mechanism are correct | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.000505 to 0.0017, margin 0.0037, SE ratio 1.0202 | pass |
| `mar_robustness` | `mechanisms_correct` | positive | the treatment and observation mechanisms are correct and the outcome regression is not | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0027 to 0.000257, margin 0.0050, SE ratio 1.0163 | pass |
| `mar_robustness` | `observation_wrong` | control | only the treatment mechanism is correct | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias -0.1975 to -0.1953, margin 0.0037, SE ratio 1.2430 | pass |
| `mar_robustness` | `outcome_correct` | positive | only the outcome regression is correct | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.000909 to 0.000462, margin 0.0023, SE ratio 3.7687 | pass |
| `mar_robustness` | `product_only` | control | the outcome regression and observation mechanism are wrong, and the product of the treatment and observation mechanisms is correct at the reference arm | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias -0.1259 to -0.1235, margin 0.0041, SE ratio 0.9745 | pass |
| `mar_robustness` | `treatment_wrong` | control | only the observation mechanism is correct | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias -0.1371 to -0.1345, margin 0.0044, SE ratio 1.9138 | pass |
| `missingness_necessity` | `paf__complete_case_control` | control | population attributable fraction: the identical estimator silently discards unobserved outcomes and ignores selection | bias interval must fall entirely outside the margin | bias -0.0592 to -0.0547, margin 0.0075 | pass |
| `missingness_necessity` | `par__complete_case_control` | control | population attributable risk: the identical estimator silently discards unobserved outcomes and ignores selection | bias interval must fall entirely outside the margin | bias -0.0304 to -0.0282, margin 0.0036 | pass |
| `missingness_necessity` | `par__declared` | positive | population attributable risk: the observation indicator is declared and every nuisance is correct | bias interval inside the equivalence margin | bias -0.0012 to 0.000973, margin 0.0037 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -7.878e-07, coverage 0.9078 to 0.9544, SE ratio 0.9591 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.000205, coverage 0.9194 to 0.9627, SE ratio 1.0021 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000069, coverage 0.9297 to 0.9698, SE ratio 1.0020 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5329 to -0.4672 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5019 to -0.4989 | pass |
| `simultaneous_coverage` | `attributable_binary__pointwise_joint_control` | control | L1: ey_obs, ey0, par and paf with MAR outcomes, in sample with the law's nuisances: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8642 to 0.8985 | pass |
| `simultaneous_coverage` | `attributable_binary__simultaneous_band` | positive | L1: ey_obs, ey0, par and paf with MAR outcomes, in sample with the law's nuisances: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9328 to 0.9570 | pass |
| `simultaneous_coverage` | `attributable_binary_cvtmle__pointwise_joint_control` | control | L1: ey_obs, ey0, par and paf with MAR outcomes, stacked CV-TMLE with depth-five trees: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8761 to 0.9090 | pass |
| `simultaneous_coverage` | `attributable_binary_cvtmle__simultaneous_band` | positive | L1: ey_obs, ey0, par and paf with MAR outcomes, stacked CV-TMLE with depth-five trees: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9402 to 0.9630 | pass |
| `simultaneous_coverage` | `attributable_clustered__pointwise_joint_control` | control | L1 rows in 100 clusters of 20 that share W: ey_obs, ey0, par and paf, in sample with id=: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8619 to 0.8965 | pass |
| `simultaneous_coverage` | `attributable_clustered__simultaneous_band` | positive | L1 rows in 100 clusters of 20 that share W: ey_obs, ey0, par and paf, in sample with id=: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9282 to 0.9533 | pass |
| `simultaneous_coverage` | `attributable_three_arm__pointwise_joint_control` | control | L3: ey_obs, the three arm means, par[low] and paf[low] with MAR outcomes, in sample: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.7894 to 0.8310 | pass |
| `simultaneous_coverage` | `attributable_three_arm__simultaneous_band` | positive | L3: ey_obs, the three arm means, par[low] and paf[low] with MAR outcomes, in sample: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9296 to 0.9544 | pass |
| `simultaneous_coverage` | `attributable_weighted__pointwise_joint_control` | control | L1 with fixed weights 0.6, 1.0 and 1.8 by W: ey_obs, ey0, par and paf, in sample: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8597 to 0.8946 | pass |
| `simultaneous_coverage` | `attributable_weighted__simultaneous_band` | positive | L1 with fixed weights 0.6, 1.0 and 1.8 by W: ey_obs, ey0, par and paf, in sample: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9365 to 0.9600 | pass |
| `targeting_necessity` | `par__targeted` | positive | population attributable risk: both fluctuations are solved; the outcome regression is wrong and both mechanisms are correct | bias interval inside the equivalence margin | bias -0.000912 to 0.0023, margin 0.0053 | pass |
| `targeting_necessity` | `par__untargeted` | control | population attributable risk: the identical wrong outcome regression is read without either fluctuation | bias interval must fall entirely outside the margin | bias -0.2283 to -0.2276, margin 0.0012 | pass |
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
| `simultaneous_coverage` | the default band of five fits, each with its pointwise joint control: in-sample L1, in-sample L3, stacked L1, and the weighted and clustered in-sample L1 fits |

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
in `tests/unit/test_simultaneous_cell_design.py`. The table gives the limiting pointwise joint
coverage of each cell. The control power is 1.0 at 2,400 replications in each cell.

| cell | covariance | pointwise joint coverage |
| --- | --- | ---: |
| `attributable_binary`, `attributable_binary_cvtmle` | exact efficient influence covariance on L1. The stacked fit has the in-sample limit | 0.8849 |
| `attributable_three_arm` | exact efficient influence covariance on L3 | 0.8148 |
| `attributable_weighted` | the tilted functional's curves under the sampling law | 0.8829 |
| `attributable_clustered` | the covariance of the cluster sums of 20 rows that share $W$ | 0.8855 |

The stacked three-arm fit creates one more band shape. It has the in-sample three-arm limit, and
the stacked binary cell measures its construction, so the gate maps it to those two cells.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 62 | truth tests passing |
| `independent_tests_total` | 62 | truth tests reported |
| `paired_tests_passed` | 31 | paired comparisons passing |
| `paired_tests_total` | 31 | paired comparisons reported |
| `property_cells_passed` | 32 | property cells passing |
| `property_cells_total` | 32 | property cells reported |
| `max_standardized_bias` | 0.0568 | largest primary standardized bias |
| `min_coverage` | 0.9350 | lowest primary coverage |
| `max_margin_utilization` | 2.718e-07 | largest paired similarity-margin share |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal test size |
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
| `margin:minimum_power` | 0.8000 | minimum power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |
| `margin:missingness_displacement` | 0.2500 | minimum complete-case displacement |
| `bound:par_standard_error` | 0.0151 | exact L1 EIF standard error of PAR at primary n |
| `bound:paf_standard_error` | 0.0321 | exact L1 EIF standard error of PAF at primary n |

## Limits

- The study covers finite-support laws with one three-level baseline covariate. The in-sample
  fits use the law's own nuisances, and the stacked fits use trees that can fit the saturated
  model.
- No cross-fitted continuous, clustered or weighted fit is admitted, so none is measured.
- Stacked strata, the bootstrap, DR-TMLE and C-TMLE are refused, so none is measured. The
  in-sample fit admits strata. `tests/unit/test_stratified_natural_course_exact.py` checks its
  stratum PAR against the stratum curves, and this study does not measure it.
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
