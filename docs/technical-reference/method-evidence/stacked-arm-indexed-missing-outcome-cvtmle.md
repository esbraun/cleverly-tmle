# Stacked arm-indexed missing-outcome CV-TMLE

This study validates the stacked CV-TMLE of arm-indexed means and contrasts under MAR.
[Stacked CV-TMLE for arm-indexed targets](../point-treatment-tmle.md#stacked-cv-tmle-for-arm-indexed-targets)
states the contract. The table gives the studied construction. Every fit declares it through the public
`TMLEMethod`, `CrossFitting`, `Targeting`, and `Inference` API.

| component | studied setting |
| --- | --- |
| partition | one package-generated, unstratified ten-fold split |
| outcome regression | fitted on the respondents in each training complement |
| treatment and response mechanisms | fitted separately on each training complement |
| targeting | one pooled logistic fluctuation with one column for each arm |
| evaluation | plug-in and influence curve over the whole sample |
| inference | pointwise Wald intervals from the centered influence-curve variance; ratios on the log scale |
| primary learners | separate depth-five trees with minimum leaf size 25 for each nuisance |

## Laws

`tests/studies/mar_arm_indexed_laws.py` declares four laws. Each law has a three-level covariate
`W`, and each nuisance is a table indexed by `W` and the arm. The smallest product `g_a pi_a` is
0.12 in L1 and L2, and 0.1125 in L3 and L4.

| law | arms | outcome | estimands |
| --- | --- | --- | --- |
| L1 | 2 | binary | `ey0`, `ey1`, `ate`, `rr`, and `or` |
| L2 | 2 | `-2 + 10 B`, with `B` Beta-distributed about the L1 mean | `ey0`, `ey1`, and `ate` |
| L3 | 3 | binary | `ey[a]` for each arm; `ate`, `rr`, and `or` against `high` |
| L4 | 3 | `-2 + 10 B`, with `B` Beta-distributed about the L3 mean | `ey[a]` for each arm; `ate` against `high` |

A continuous fit declares `q_bounds=(-2, 8)`, the known support. Each law has nine or fewer
covariate cells for the outcome regression. Each depth-five tree can therefore fit the saturated
model, and that model is correct. The primary and calibration rows thus do not test calibration
under a misspecified data-adaptive learner. The paired overfitting cells supply that evidence.

## Comparator

The study compares R `tmle` 2.1.1. The adapter supplies the same stitched out-of-fold outcome,
treatment, and response predictions that `cleverly` targets. L1 and L2 use the native two-arm
path. L3 and L4 use the population-mean path once for each arm. For arm `a`, the adapter passes a
constant `A`, the indicator `1{A = a} Delta` as `Delta`, and `NA` for every other outcome. It
builds the joint covariance and each reference contrast from the three returned `IC.EY1` curves.

R takes a continuous outcome scale from every non-`NA` `Y` (`tmle.R` line 1120). A continuous fit
therefore sets `Y` to `-2` and `8` on two rows whose response indicator is zero. Those rows leave
the fluctuation and the curves. `probe_scale_workaround.R` checks the workaround on replication 0
of each continuous law, and it writes `scale-probe.csv`.

| probe check | requirement |
| --- | --- |
| scale with the planted rows | equal to `q_bounds` exactly |
| scale without the planted rows | different from `q_bounds`, so the workaround does work |
| planted rows moved to two other eligible rows | every point and curve value bitwise unchanged |
| independent rebuild at the `q_bounds` scale | points and curves within `1e-12` |

The regeneration refuses publication when a probe row fails. Every committed probe row passes.

This comparison conditions on the supplied nuisance predictions. It validates the pooled
targeting, plug-in, influence curves, and joint covariance. It does not validate R-native
cross-fit training. [`docs/references.md`](../../references.md) records the locator for each
candidate below.

The paired tests alone cannot show that `cleverly` targets. The saturated trees leave the initial
plug-in almost unbiased, so an estimate without its targeting step still passes every paired test
and every truth test. The claim that `cleverly` reproduces the R targeting step rests on a separate
check over the committed replications.

| check | requirement |
| --- | --- |
| targeting move, `estimate - initial_estimate` | nonzero in every replication and estimand |
| `abs(cleverly - R) / abs(targeting move)` | below `1e-2` in every replication and estimand |
| initial estimates of the two implementations | equal to `1e-12` |
| mutation: `cleverly` reports its initial estimate | the ratio exceeds `0.9` in every row, so the check fails |

R `tmle` stops its fluctuation `glm` at a relative deviance change of `1e-8`. That stop resolves
the fluctuation coefficient to about `1e-4` of its scale, and the bound allows a hundredfold over
that. `tests/unit/test_arm_indexed_cvtmle_method_study.py` runs the check and the mutation.

| candidate | disposition |
| --- | --- |
| R `tmle` 2.1.1 | compared through its two-arm path for L1 and L2, and its population-mean path for each arm of L3 and L4 |
| `tmle3` at commit `ed72f8a` | its generic treatment-specific outcome fit can use the `Delta = 0` pseudo-outcomes when it predicts under `Delta = 1`. `cleverly` fits that regression on respondents only |
| zEpid 0.9.1 | its cross-fit TMLE targets inside each fold rather than with one pooled fluctuation |
| Newey and Robins (2018) | the construction fits the outcome and inverse response regressions on distinct subsamples. That is a different estimator |

| source | locator | what it supplies |
| --- | --- | --- |
| Díaz and van der Laan (2017) | Section 2.1 and Equation (1) | the per-arm curve with missing outcomes |
| Gruber and van der Laan (2012) | Section 2.3 | the clever covariate for a generic arm with a treatment mechanism conditional on `W` |
| Gruber and van der Laan (2012) | Appendix A | the delta-method form of each contrast |
| Zheng and van der Laan (2011) | Section 2 and Theorem 2 | the vector parameter, the training-complement fits, and the joint expansion |
| Levy (2018) | Section 3.1 | the stacked update and the whole-sample plug-in |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L3: three-arm binary-outcome law with MAR outcomes | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0041 to 0.0027 | 0.9550 | 1.0352 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `ate[low vs high]` | difference in counterfactual means, low versus high | R `tmle` two-arm path, or its population-mean path once per arm | -0.0041 to 0.0027 | 0.9550 | 1.0352 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `ate[mid vs high]` | difference in counterfactual means, mid versus high | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0044 to 0.0021 | 0.9613 | 1.0268 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `ate[mid vs high]` | difference in counterfactual means, mid versus high | R `tmle` two-arm path, or its population-mean path once per arm | -0.0044 to 0.0021 | 0.9613 | 1.0268 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0012 to 0.0036 | 0.9575 | 1.0236 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `ey[high]` | counterfactual mean under treatment arm 'high' | R `tmle` two-arm path, or its population-mean path once per arm | -0.0012 to 0.0036 | 0.9575 | 1.0236 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0020 to 0.0029 | 0.9450 | 1.0318 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `ey[low]` | counterfactual mean under treatment arm 'low' | R `tmle` two-arm path, or its population-mean path once per arm | -0.0020 to 0.0029 | 0.9450 | 1.0318 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `ey[mid]` | counterfactual mean under treatment arm 'mid' | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0022 to 0.0022 | 0.9437 | 1.0215 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `ey[mid]` | counterfactual mean under treatment arm 'mid' | R `tmle` two-arm path, or its population-mean path once per arm | -0.0022 to 0.0022 | 0.9437 | 1.0215 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0229 to 0.0074 | 0.9613 | 1.0329 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | R `tmle` two-arm path, or its population-mean path once per arm | -0.0229 to 0.0074 | 0.9613 | 1.0329 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `or[mid vs high]` | marginal odds ratio, mid versus high, reported on the log scale | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0223 to 0.0065 | 0.9613 | 1.0244 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `or[mid vs high]` | marginal odds ratio, mid versus high, reported on the log scale | R `tmle` two-arm path, or its population-mean path once per arm | -0.0223 to 0.0065 | 0.9613 | 1.0244 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0095 to 0.0050 | 0.9537 | 1.0360 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | R `tmle` two-arm path, or its population-mean path once per arm | -0.0095 to 0.0050 | 0.9537 | 1.0360 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `rr[mid vs high]` | marginal risk ratio, mid versus high, reported on the log scale | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0073 to 0.0034 | 0.9550 | 1.0305 | pass |
| L3: three-arm binary-outcome law with MAR outcomes | `rr[mid vs high]` | marginal risk ratio, mid versus high, reported on the log scale | R `tmle` two-arm path, or its population-mean path once per arm | -0.0073 to 0.0034 | 0.9550 | 1.0305 | pass |
| L1: two-arm binary-outcome law with MAR outcomes | `ate` | average treatment effect | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0037 to 0.0023 | 0.9500 | 1.0345 | pass |
| L1: two-arm binary-outcome law with MAR outcomes | `ate` | average treatment effect | R `tmle` two-arm path, or its population-mean path once per arm | -0.0037 to 0.0023 | 0.9500 | 1.0345 | pass |
| L1: two-arm binary-outcome law with MAR outcomes | `ey0` | counterfactual mean under no treatment | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0017 to 0.0028 | 0.9437 | 0.9731 | pass |
| L1: two-arm binary-outcome law with MAR outcomes | `ey0` | counterfactual mean under no treatment | R `tmle` two-arm path, or its population-mean path once per arm | -0.0017 to 0.0028 | 0.9437 | 0.9731 | pass |
| L1: two-arm binary-outcome law with MAR outcomes | `ey1` | counterfactual mean under treatment | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0023 to 0.0020 | 0.9425 | 1.0062 | pass |
| L1: two-arm binary-outcome law with MAR outcomes | `ey1` | counterfactual mean under treatment | R `tmle` two-arm path, or its population-mean path once per arm | -0.0023 to 0.0020 | 0.9425 | 1.0062 | pass |
| L1: two-arm binary-outcome law with MAR outcomes | `or` | marginal odds ratio, reported on the log scale | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0132 to 0.0119 | 0.9525 | 1.0341 | pass |
| L1: two-arm binary-outcome law with MAR outcomes | `or` | marginal odds ratio, reported on the log scale | R `tmle` two-arm path, or its population-mean path once per arm | -0.0132 to 0.0119 | 0.9525 | 1.0341 | pass |
| L1: two-arm binary-outcome law with MAR outcomes | `rr` | marginal risk ratio, reported on the log scale | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0070 to 0.0064 | 0.9500 | 1.0230 | pass |
| L1: two-arm binary-outcome law with MAR outcomes | `rr` | marginal risk ratio, reported on the log scale | R `tmle` two-arm path, or its population-mean path once per arm | -0.0070 to 0.0064 | 0.9500 | 1.0230 | pass |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0237 to 0.0095 | 0.9387 | 0.9653 | pass |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ate[low vs high]` | difference in counterfactual means, low versus high | R `tmle` two-arm path, or its population-mean path once per arm | -0.0237 to 0.0095 | 0.9387 | 0.9653 | pass |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ate[mid vs high]` | difference in counterfactual means, mid versus high | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0148 to 0.0180 | 0.9475 | 0.9939 | pass |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ate[mid vs high]` | difference in counterfactual means, mid versus high | R `tmle` two-arm path, or its population-mean path once per arm | -0.0148 to 0.0180 | 0.9475 | 0.9939 | pass |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0056 to 0.0171 | 0.9425 | 0.9927 | pass |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ey[high]` | counterfactual mean under treatment arm 'high' | R `tmle` two-arm path, or its population-mean path once per arm | -0.0056 to 0.0171 | 0.9425 | 0.9927 | pass |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0133 to 0.0106 | 0.9300 | 0.9565 | pass |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ey[low]` | counterfactual mean under treatment arm 'low' | R `tmle` two-arm path, or its population-mean path once per arm | -0.0133 to 0.0106 | 0.9300 | 0.9565 | pass |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ey[mid]` | counterfactual mean under treatment arm 'mid' | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0032 to 0.0178 | 0.9487 | 1.0190 | pass |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ey[mid]` | counterfactual mean under treatment arm 'mid' | R `tmle` two-arm path, or its population-mean path once per arm | -0.0032 to 0.0178 | 0.9487 | 1.0190 | pass |
| L2: two-arm bounded continuous-outcome law with MAR outcomes | `ate` | average treatment effect | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0114 to 0.0191 | 0.9413 | 0.9797 | pass |
| L2: two-arm bounded continuous-outcome law with MAR outcomes | `ate` | average treatment effect | R `tmle` two-arm path, or its population-mean path once per arm | -0.0114 to 0.0191 | 0.9413 | 0.9797 | pass |
| L2: two-arm bounded continuous-outcome law with MAR outcomes | `ey0` | counterfactual mean under no treatment | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0142 to 0.0057 | 0.9537 | 1.0180 | pass |
| L2: two-arm bounded continuous-outcome law with MAR outcomes | `ey0` | counterfactual mean under no treatment | R `tmle` two-arm path, or its population-mean path once per arm | -0.0142 to 0.0057 | 0.9537 | 1.0180 | pass |
| L2: two-arm bounded continuous-outcome law with MAR outcomes | `ey1` | counterfactual mean under treatment | `cleverly` stacked arm-indexed missing-outcome CV-TMLE | -0.0112 to 0.0104 | 0.9487 | 0.9806 | pass |
| L2: two-arm bounded continuous-outcome law with MAR outcomes | `ey1` | counterfactual mean under treatment | R `tmle` two-arm path, or its population-mean path once per arm | -0.0112 to 0.0104 | 0.9487 | 0.9806 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L3: three-arm binary-outcome law with MAR outcomes | `ate[low vs high]` | difference in counterfactual means, low versus high | -4.105e-10 | 7.373e-08 | 1.0000 | 0 | 1.793e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes | `ate[mid vs high]` | difference in counterfactual means, mid versus high | -1.973e-10 | 3.676e-08 | 1.0000 | 0 | 2.471e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes | `ey[high]` | counterfactual mean under treatment arm 'high' | 2.215e-10 | 5.617e-08 | 1.0000 | 0 | 3.387e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes | `ey[low]` | counterfactual mean under treatment arm 'low' | -1.890e-10 | 4.745e-08 | 1.0000 | 0 | 1.410e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes | `ey[mid]` | counterfactual mean under treatment arm 'mid' | 2.416e-11 | 6.736e-09 | 1.0000 | 0 | 1.773e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | -5.201e-10 | 7.413e-08 | 1.0000 | 0 | 2.123e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes | `or[mid vs high]` | marginal odds ratio, mid versus high, reported on the log scale | -4.952e-10 | 3.866e-08 | 1.0000 | 0 | 2.855e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | -4.554e-10 | 6.914e-08 | 1.0000 | 0 | 1.803e-09 vs 0.0500 | equivalent |
| L3: three-arm binary-outcome law with MAR outcomes | `rr[mid vs high]` | marginal risk ratio, mid versus high, reported on the log scale | -2.176e-10 | 3.121e-08 | 1.0000 | 0 | 2.082e-09 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome law with MAR outcomes | `ate` | average treatment effect | 9.460e-10 | 1.924e-07 | 1.0000 | 0 | 4.979e-09 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome law with MAR outcomes | `ey0` | counterfactual mean under no treatment | -3.362e-10 | 9.074e-08 | 1.0000 | 0 | 4.969e-09 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome law with MAR outcomes | `ey1` | counterfactual mean under treatment | 6.099e-10 | 1.727e-07 | 1.0000 | 0 | 4.987e-09 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome law with MAR outcomes | `or` | marginal odds ratio, reported on the log scale | 8.961e-09 | 1.837e-07 | 1.0000 | 0 | 4.783e-09 vs 0.0500 | equivalent |
| L1: two-arm binary-outcome law with MAR outcomes | `rr` | marginal risk ratio, reported on the log scale | 2.948e-09 | 1.713e-07 | 1.0000 | 0 | 4.019e-09 vs 0.0500 | equivalent |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ate[low vs high]` | difference in counterfactual means, low versus high | -5.692e-10 | 2.083e-08 | 1.0000 | 0 | 1.011e-09 vs 0.0500 | equivalent |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ate[mid vs high]` | difference in counterfactual means, mid versus high | 3.131e-10 | 1.162e-08 | 1.0000 | 0 | 1.915e-10 vs 0.0500 | equivalent |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ey[high]` | counterfactual mean under treatment arm 'high' | -4.417e-13 | 2.375e-11 | 1.0000 | 0 | 5.510e-13 vs 0.0500 | equivalent |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ey[low]` | counterfactual mean under treatment arm 'low' | -5.697e-10 | 2.898e-08 | 1.0000 | 0 | 1.762e-09 vs 0.0500 | equivalent |
| L4: three-arm bounded continuous-outcome law with MAR outcomes | `ey[mid]` | counterfactual mean under treatment arm 'mid' | 3.126e-10 | 1.808e-08 | 1.0000 | 0 | 4.701e-09 vs 0.0500 | equivalent |
| L2: two-arm bounded continuous-outcome law with MAR outcomes | `ate` | average treatment effect | 1.598e-09 | 6.375e-08 | 1.0000 | 0 | 1.537e-08 vs 0.0500 | equivalent |
| L2: two-arm bounded continuous-outcome law with MAR outcomes | `ey0` | counterfactual mean under no treatment | -7.787e-10 | 4.754e-08 | 1.0000 | 0 | 9.779e-10 vs 0.0500 | equivalent |
| L2: two-arm bounded continuous-outcome law with MAR outcomes | `ey1` | counterfactual mean under treatment | 8.193e-10 | 4.628e-08 | 1.0000 | 0 | 7.143e-09 vs 0.0500 | equivalent |
<!-- /generated -->

Both implementations use the centered, `n - 1` divisor variance `var(IC) / n`. The R two-arm path
bounds a binary mean interval to `[0, 1]` and an ATE interval to `[-1, 1]`. These laws stay away
from those bounds. For L3 and L4 the adapter forms Wald intervals from the per-arm curves.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `l1_ate__in_sample_control` | control | L1 average treatment effect: the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.1100 to 0.1371 | pass |
| `crossfit_overfitting` | `l1_ate__stacked_arm_indexed_cvtmle` | positive | L1 average treatment effect: stacked arm-indexed MAR CV-TMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.8724 to 1.0501 | pass |
| `crossfit_overfitting` | `l2_ate__in_sample_control` | control | L2 average treatment effect: the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.1666 to 0.2116 | pass |
| `crossfit_overfitting` | `l2_ate__stacked_arm_indexed_cvtmle` | positive | L2 average treatment effect: stacked arm-indexed MAR CV-TMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.8550 to 1.0245 | pass |
| `crossfit_overfitting` | `l3_ate_low__in_sample_control` | control | L3 difference, low versus high: the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.1454 to 0.1820 | pass |
| `crossfit_overfitting` | `l3_ate_low__stacked_arm_indexed_cvtmle` | positive | L3 difference, low versus high: stacked arm-indexed MAR CV-TMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9195 to 1.1130 | pass |
| `crossfit_overfitting` | `l4_ate_low__in_sample_control` | control | L4 difference, low versus high: the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.2333 to 0.3088 | pass |
| `crossfit_overfitting` | `l4_ate_low__stacked_arm_indexed_cvtmle` | positive | L4 difference, low versus high: stacked arm-indexed MAR CV-TMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.8759 to 1.0735 | pass |
| `interval_calibration` | `l1_ate__learned_nuisances` | positive | L1 average treatment effect: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9361 to 0.9617, SE ratio 0.9413 to 1.0198 | pass |
| `interval_calibration` | `l1_ate__shrunken_se_control` | control | L1 average treatment effect: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8058 to 0.8497, SE ratio 0.6594 to 0.7139 | pass |
| `interval_calibration` | `l1_ey0__learned_nuisances` | positive | L1 mean under arm 0: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9311 to 0.9578, SE ratio 0.9465 to 1.0261 | pass |
| `interval_calibration` | `l1_ey0__shrunken_se_control` | control | L1 mean under arm 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7943 to 0.8392, SE ratio 0.6620 to 0.7170 | pass |
| `interval_calibration` | `l1_ey1__learned_nuisances` | positive | L1 mean under arm 1: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9316 to 0.9582, SE ratio 0.9549 to 1.0399 | pass |
| `interval_calibration` | `l1_ey1__shrunken_se_control` | control | L1 mean under arm 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7953 to 0.8402, SE ratio 0.6681 to 0.7282 | pass |
| `interval_calibration` | `l1_or__learned_nuisances` | positive | L1 log odds ratio: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9361 to 0.9617, SE ratio 0.9416 to 1.0195 | pass |
| `interval_calibration` | `l1_or__shrunken_se_control` | control | L1 log odds ratio: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8047 to 0.8487, SE ratio 0.6596 to 0.7136 | pass |
| `interval_calibration` | `l1_rr__learned_nuisances` | positive | L1 log risk ratio: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9322 to 0.9586, SE ratio 0.9396 to 1.0186 | pass |
| `interval_calibration` | `l1_rr__shrunken_se_control` | control | L1 log risk ratio: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8005 to 0.8449, SE ratio 0.6581 to 0.7130 | pass |
| `interval_calibration` | `l2_ate__learned_nuisances` | positive | L2 average treatment effect: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9383 to 0.9635, SE ratio 0.9684 to 1.0485 | pass |
| `interval_calibration` | `l2_ate__shrunken_se_control` | control | L2 average treatment effect: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8131 to 0.8563, SE ratio 0.6786 to 0.7348 | pass |
| `interval_calibration` | `l2_ey0__learned_nuisances` | positive | L2 mean under arm 0: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9428 to 0.9670, SE ratio 0.9800 to 1.0614 | pass |
| `interval_calibration` | `l2_ey0__shrunken_se_control` | control | L2 mean under arm 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8152 to 0.8582, SE ratio 0.6855 to 0.7429 | pass |
| `interval_calibration` | `l2_ey1__learned_nuisances` | positive | L2 mean under arm 1: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9422 to 0.9665, SE ratio 0.9806 to 1.0627 | pass |
| `interval_calibration` | `l2_ey1__shrunken_se_control` | control | L2 mean under arm 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8173 to 0.8601, SE ratio 0.6862 to 0.7442 | pass |
| `interval_calibration` | `l3_ate_low__learned_nuisances` | positive | L3 difference, low versus high: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9383 to 0.9635, SE ratio 0.9835 to 1.0666 | pass |
| `interval_calibration` | `l3_ate_low__shrunken_se_control` | control | L3 difference, low versus high: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8162 to 0.8591, SE ratio 0.6877 to 0.7468 | pass |
| `interval_calibration` | `l3_ate_mid__learned_nuisances` | positive | L3 difference, mid versus high: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9406 to 0.9652, SE ratio 0.9718 to 1.0570 | pass |
| `interval_calibration` | `l3_ate_mid__shrunken_se_control` | control | L3 difference, mid versus high: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8110 to 0.8544, SE ratio 0.6798 to 0.7413 | pass |
| `interval_calibration` | `l3_ey_high__learned_nuisances` | positive | L3 mean under arm high: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9294 to 0.9564, SE ratio 0.9517 to 1.0417 | pass |
| `interval_calibration` | `l3_ey_high__shrunken_se_control` | control | L3 mean under arm high: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8068 to 0.8506, SE ratio 0.6676 to 0.7280 | pass |
| `interval_calibration` | `l3_ey_low__learned_nuisances` | positive | L3 mean under arm low: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9372 to 0.9626, SE ratio 0.9820 to 1.0616 | pass |
| `interval_calibration` | `l3_ey_low__shrunken_se_control` | control | L3 mean under arm low: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8173 to 0.8601, SE ratio 0.6880 to 0.7442 | pass |
| `interval_calibration` | `l3_ey_mid__learned_nuisances` | positive | L3 mean under arm mid: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9394 to 0.9643, SE ratio 0.9724 to 1.0565 | pass |
| `interval_calibration` | `l3_ey_mid__shrunken_se_control` | control | L3 mean under arm mid: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8168 to 0.8596, SE ratio 0.6808 to 0.7400 | pass |
| `interval_calibration` | `l3_or_low__learned_nuisances` | positive | L3 log odds ratio, low versus high: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9389 to 0.9639, SE ratio 0.9826 to 1.0648 | pass |
| `interval_calibration` | `l3_or_low__shrunken_se_control` | control | L3 log odds ratio, low versus high: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8152 to 0.8582, SE ratio 0.6883 to 0.7452 | pass |
| `interval_calibration` | `l3_or_mid__learned_nuisances` | positive | L3 log odds ratio, mid versus high: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9445 to 0.9683, SE ratio 0.9693 to 1.0571 | pass |
| `interval_calibration` | `l3_or_mid__shrunken_se_control` | control | L3 log odds ratio, mid versus high: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8100 to 0.8534, SE ratio 0.6789 to 0.7411 | pass |
| `interval_calibration` | `l3_rr_low__learned_nuisances` | positive | L3 log risk ratio, low versus high: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9456 to 0.9691, SE ratio 0.9877 to 1.0688 | pass |
| `interval_calibration` | `l3_rr_low__shrunken_se_control` | control | L3 log risk ratio, low versus high: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8194 to 0.8619, SE ratio 0.6928 to 0.7493 | pass |
| `interval_calibration` | `l3_rr_mid__learned_nuisances` | positive | L3 log risk ratio, mid versus high: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9411 to 0.9657, SE ratio 0.9738 to 1.0627 | pass |
| `interval_calibration` | `l3_rr_mid__shrunken_se_control` | control | L3 log risk ratio, mid versus high: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8189 to 0.8615, SE ratio 0.6823 to 0.7432 | pass |
| `interval_calibration` | `l4_ate_low__learned_nuisances` | positive | L4 difference, low versus high: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9344 to 0.9604, SE ratio 0.9672 to 1.0461 | pass |
| `interval_calibration` | `l4_ate_low__shrunken_se_control` | control | L4 difference, low versus high: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8021 to 0.8463, SE ratio 0.6763 to 0.7331 | pass |
| `interval_calibration` | `l4_ate_mid__learned_nuisances` | positive | L4 difference, mid versus high: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9394 to 0.9643, SE ratio 0.9807 to 1.0590 | pass |
| `interval_calibration` | `l4_ate_mid__shrunken_se_control` | control | L4 difference, mid versus high: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8126 to 0.8558, SE ratio 0.6865 to 0.7419 | pass |
| `interval_calibration` | `l4_ey_high__learned_nuisances` | positive | L4 mean under arm high: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9316 to 0.9582, SE ratio 0.9740 to 1.0564 | pass |
| `interval_calibration` | `l4_ey_high__shrunken_se_control` | control | L4 mean under arm high: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8079 to 0.8515, SE ratio 0.6823 to 0.7382 | pass |
| `interval_calibration` | `l4_ey_low__learned_nuisances` | positive | L4 mean under arm low: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9327 to 0.9591, SE ratio 0.9625 to 1.0433 | pass |
| `interval_calibration` | `l4_ey_low__shrunken_se_control` | control | L4 mean under arm low: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8162 to 0.8591, SE ratio 0.6739 to 0.7320 | pass |
| `interval_calibration` | `l4_ey_mid__learned_nuisances` | positive | L4 mean under arm mid: separate depth-five trees fit the outcome, treatment and observation nuisances out of fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9411 to 0.9657, SE ratio 0.9783 to 1.0615 | pass |
| `interval_calibration` | `l4_ey_mid__shrunken_se_control` | control | L4 mean under arm mid: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8126 to 0.8558, SE ratio 0.6854 to 0.7427 | pass |
| `mar_robustness` | `l1_ate__only_outcome_wrong` | positive | L1 average treatment effect: only the outcome regression is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0036 to 0.0025, margin 0.0101, SE ratio 0.9953 | pass |
| `mar_robustness` | `l1_ate__only_response_wrong` | positive | L1 average treatment effect: only the observation mechanism is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0026 to 0.0010, margin 0.0060, SE ratio 1.6416 | pass |
| `mar_robustness` | `l1_ate__only_treatment_wrong` | positive | L1 average treatment effect: only the treatment mechanism is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0025 to 0.0020, margin 0.0075, SE ratio 1.4196 | pass |
| `mar_robustness` | `l1_ate__outcome_and_response_wrong` | control | L1 average treatment effect: the outcome regression and the observation mechanism are both wrong | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias -0.4808 to -0.4769, margin 0.0066, SE ratio 1.7487 | pass |
| `mar_robustness` | `l2_ate__only_outcome_wrong` | positive | L2 average treatment effect: only the outcome regression is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0437 to -0.0013, margin 0.0711, SE ratio 0.9797 | pass |
| `mar_robustness` | `l2_ate__only_response_wrong` | positive | L2 average treatment effect: only the observation mechanism is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0174 to 0.0015, margin 0.0316, SE ratio 1.4776 | pass |
| `mar_robustness` | `l2_ate__only_treatment_wrong` | positive | L2 average treatment effect: only the treatment mechanism is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0074 to 0.0154, margin 0.0382, SE ratio 1.3065 | pass |
| `mar_robustness` | `l2_ate__outcome_and_response_wrong` | control | L2 average treatment effect: the outcome regression and the observation mechanism are both wrong | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias -4.8100 to -4.7846, margin 0.0425, SE ratio 1.7963 | pass |
| `mar_robustness` | `l3_ate_low__only_outcome_wrong` | positive | L3 difference, low versus high: only the outcome regression is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0039 to 0.0025, margin 0.0109, SE ratio 0.9868 | pass |
| `mar_robustness` | `l3_ate_low__only_response_wrong` | positive | L3 difference, low versus high: only the observation mechanism is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0027 to 0.0022, margin 0.0082, SE ratio 1.7628 | pass |
| `mar_robustness` | `l3_ate_low__only_treatment_wrong` | positive | L3 difference, low versus high: only the treatment mechanism is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0021 to 0.0026, margin 0.0079, SE ratio 1.4943 | pass |
| `mar_robustness` | `l3_ate_low__outcome_and_response_wrong` | control | L3 difference, low versus high: the outcome regression and the observation mechanism are both wrong | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias 0.2911 to 0.2969, margin 0.0098, SE ratio 1.7518 | pass |
| `mar_robustness` | `l3_ate_mid__only_outcome_wrong` | positive | L3 difference, mid versus high: only the outcome regression is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0035 to 0.0029, margin 0.0108, SE ratio 0.9779 | pass |
| `mar_robustness` | `l3_ate_mid__only_response_wrong` | positive | L3 difference, mid versus high: only the observation mechanism is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0030 to 0.0021, margin 0.0086, SE ratio 1.5259 | pass |
| `mar_robustness` | `l3_ate_mid__only_treatment_wrong` | positive | L3 difference, mid versus high: only the treatment mechanism is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0030 to 0.0012, margin 0.0072, SE ratio 1.3532 | pass |
| `mar_robustness` | `l3_ate_mid__outcome_and_response_wrong` | control | L3 difference, mid versus high: the outcome regression and the observation mechanism are both wrong | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias 0.1499 to 0.1558, margin 0.0100, SE ratio 1.4921 | pass |
| `mar_robustness` | `l4_ate_low__only_outcome_wrong` | positive | L4 difference, low versus high: only the outcome regression is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0219 to 0.0187, margin 0.0681, SE ratio 0.9865 | pass |
| `mar_robustness` | `l4_ate_low__only_response_wrong` | positive | L4 difference, low versus high: only the observation mechanism is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0130 to 0.0087, margin 0.0365, SE ratio 1.7824 | pass |
| `mar_robustness` | `l4_ate_low__only_treatment_wrong` | positive | L4 difference, low versus high: only the treatment mechanism is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0183 to 0.0030, margin 0.0359, SE ratio 1.4934 | pass |
| `mar_robustness` | `l4_ate_low__outcome_and_response_wrong` | control | L4 difference, low versus high: the outcome regression and the observation mechanism are both wrong | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias 2.9086 to 2.9477, margin 0.0656, SE ratio 1.7332 | pass |
| `mar_robustness` | `l4_ate_mid__only_outcome_wrong` | positive | L4 difference, mid versus high: only the outcome regression is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0101 to 0.0313, margin 0.0694, SE ratio 1.0074 | pass |
| `mar_robustness` | `l4_ate_mid__only_response_wrong` | positive | L4 difference, mid versus high: only the observation mechanism is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0238 to 0.0019, margin 0.0432, SE ratio 1.4191 | pass |
| `mar_robustness` | `l4_ate_mid__only_treatment_wrong` | positive | L4 difference, mid versus high: only the treatment mechanism is wrong | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0122 to 0.0101, margin 0.0373, SE ratio 1.2520 | pass |
| `mar_robustness` | `l4_ate_mid__outcome_and_response_wrong` | control | L4 difference, mid versus high: the outcome regression and the observation mechanism are both wrong | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias 1.4891 to 1.5278, margin 0.0648, SE ratio 1.4437 | pass |
| `simultaneous_coverage` | `l1__pointwise_joint_control` | control | L1, two arms and a binary outcome: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8575 to 0.8957 | pass |
| `simultaneous_coverage` | `l1__simultaneous_band` | positive | L1, two arms and a binary outcome: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9255 to 0.9533 | pass |
| `simultaneous_coverage` | `l2__pointwise_joint_control` | control | L2, two arms and a bounded continuous outcome: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8761 to 0.9119 | pass |
| `simultaneous_coverage` | `l2__simultaneous_band` | positive | L2, two arms and a bounded continuous outcome: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9422 to 0.9665 | pass |
| `simultaneous_coverage` | `l3__pointwise_joint_control` | control | L3, three arms and a binary outcome: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8047 to 0.8487 | pass |
| `simultaneous_coverage` | `l3__simultaneous_band` | positive | L3, three arms and a binary outcome: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9344 to 0.9604 | pass |
| `simultaneous_coverage` | `l4__pointwise_joint_control` | control | L4, three arms and a bounded continuous outcome: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8021 to 0.8463 | pass |
| `simultaneous_coverage` | `l4__simultaneous_band` | positive | L4, three arms and a bounded continuous outcome: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9372 to 0.9626 | pass |
<!-- /generated -->

The table below states what each family checks.

| family | fits | cells |
| --- | --- | --- |
| `interval_calibration` | 2,000 samples of 2,000 rows for each law, with the primary trees | one positive cell for each of the 22 estimands, and a control that multiplies each standard error by `margin:shrunken_se_factor` |
| `simultaneous_coverage` | the same fits, which also declare `Inference(simultaneous=True)` | the default multiplier band over each law's estimands, and a control that reads the same fits' pointwise intervals jointly |
| `mar_robustness` | 1,200 samples of 2,000 rows for each law and configuration, with exact or deliberately wrong nuisance tables | only `Q`, only `g`, or only `pi` wrong for each ATE contrast, and a control with `Q` and `pi` both wrong |
| `crossfit_overfitting` | 400 samples of 500 rows for each law, with eight noise covariates | a fully grown outcome tree beside the exact mechanisms, fitted out of fold and in sample on the same draws |

A ratio row is on the log scale, which is the scale its interval uses. A joint cell has no scalar
estimand. Its row records the largest standardized deviation over the law's estimands, so the
table quotes its joint coverage alone.

The pointwise joint control is designed to under-cover. In the Gaussian limit, the pointwise
intervals of one law cover every truth at once in less than 90% of samples
(`tests/unit/test_arm_indexed_cvtmle_method_study.py`). The band cell therefore shows that the
band's wider critical value restores the nominal joint rate.

The control cannot fail a wrong band. A Šidák or Bonferroni band ignores the correlation between
the estimands. Each one is wider than the max-t band and passes three of the four coverage cells.
A second check reads the committed critical values of each law.

| check | requirement |
| --- | --- |
| target | the limiting max-t quantile, from each law's exact influence-curve correlation, at the level that `np.quantile` of 1,000 multiplier draws reaches on average (0.94910) |
| mean committed critical value | within `0.005` of the target for each law |
| mutation: a Šidák or a Bonferroni critical value | at least `0.08` from the target for each law, so the check fails |

Each critical value has a Monte Carlo standard deviation near 0.054 from its 1,000 draws. The
tolerance is four standard errors of a mean over 2,000 replications.
`tests/unit/test_arm_indexed_cvtmle_method_study.py` runs the check and both mutations.

A wrong outcome regression is `1 - mu` on the `[0, 1]` scale. Each wrong mechanism is a fixed
table that `tests/studies/mar_arm_indexed_laws.py` declares. The large-sample limit of each
single-wrong fit equals the truth to `1e-12`. The limit of the control moves each ATE contrast by
more than a tenth of the outcome range. The unit tests above check both statements.

The four overfitting pairs share one resampling label for their coverage-gain intervals. Their
bootstrap draws are therefore correlated across the pairs.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 44 | truth tests passing |
| `independent_tests_total` | 44 | truth tests reported |
| `paired_tests_passed` | 22 | paired comparisons passing |
| `paired_tests_total` | 22 | paired comparisons reported |
| `property_cells_passed` | 84 | property cells passing |
| `property_cells_total` | 84 | property cells reported |
| `max_standardized_bias` | 0.0636 | largest primary standardized bias |
| `min_coverage` | 0.9300 | lowest primary coverage |
| `max_margin_utilization` | 1.924e-07 | largest share of the paired margin used |
| `properties[simultaneous_coverage/l1__simultaneous_band]:coverage` | 0.9405 | L1 band joint coverage |
| `properties[simultaneous_coverage/l2__simultaneous_band]:coverage` | 0.9555 | L2 band joint coverage |
| `properties[simultaneous_coverage/l3__simultaneous_band]:coverage` | 0.9485 | L3 band joint coverage |
| `properties[simultaneous_coverage/l4__simultaneous_band]:coverage` | 0.9510 | L4 band joint coverage |
| `properties[simultaneous_coverage/l3__pointwise_joint_control]:coverage` | 0.8275 | L3 pointwise joint coverage |
| `properties[mar_robustness/l1_ate__outcome_and_response_wrong]:bias` | -0.4788 | L1 ATE bias of the control |
| `properties[mar_robustness/l3_ate_mid__outcome_and_response_wrong]:bias` | 0.1529 | L3 mid-versus-high bias of the control |
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
| `margin:calibration_coverage_lower` | 0.9200 | calibration and band coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration and band coverage upper bound |
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
| `margin:overfit_se_floor` | 0.8500 | cross-fit SE-ratio floor |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample SE-ratio ceiling |
| `margin:overfit_coverage_gain` | 0.1500 | paired coverage-gain floor |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `bound:l1_ey0_standard_error` | 0.0238 | exact EIF standard error at primary n |
| `bound:l1_ey1_standard_error` | 0.0233 | exact EIF standard error at primary n |
| `bound:l1_ate_standard_error` | 0.0335 | exact EIF standard error at primary n |
| `bound:l1_rr_standard_error` | 0.0743 | exact EIF standard error at primary n |
| `bound:l1_or_standard_error` | 0.1403 | exact EIF standard error at primary n |
| `bound:l2_ey0_standard_error` | 0.1100 | exact EIF standard error at primary n |
| `bound:l2_ey1_standard_error` | 0.1144 | exact EIF standard error at primary n |
| `bound:l2_ate_standard_error` | 0.1619 | exact EIF standard error at primary n |
| `bound:l3_ey_high_standard_error` | 0.0265 | exact EIF standard error at primary n |
| `bound:l3_ey_low_standard_error` | 0.0269 | exact EIF standard error at primary n |
| `bound:l3_ey_mid_standard_error` | 0.0241 | exact EIF standard error at primary n |
| `bound:l3_ate_low_standard_error` | 0.0378 | exact EIF standard error at primary n |
| `bound:l3_ate_mid_standard_error` | 0.0362 | exact EIF standard error at primary n |
| `bound:l3_rr_low_standard_error` | 0.0807 | exact EIF standard error at primary n |
| `bound:l3_rr_mid_standard_error` | 0.0596 | exact EIF standard error at primary n |
| `bound:l3_or_low_standard_error` | 0.1677 | exact EIF standard error at primary n |
| `bound:l3_or_mid_standard_error` | 0.1580 | exact EIF standard error at primary n |
| `bound:l4_ey_high_standard_error` | 0.1214 | exact EIF standard error at primary n |
| `bound:l4_ey_low_standard_error` | 0.1237 | exact EIF standard error at primary n |
| `bound:l4_ey_mid_standard_error` | 0.1156 | exact EIF standard error at primary n |
| `bound:l4_ate_low_standard_error` | 0.1734 | exact EIF standard error at primary n |
| `bound:l4_ate_mid_standard_error` | 0.1760 | exact EIF standard error at primary n |

## Limits

- The evidence covers four finite laws with one three-level baseline covariate. Two laws have two
  arms and two have three. Two have a binary outcome and two a Beta outcome on a known support.
- It covers one package-generated unstratified ten-fold draw, pooled targeting, whole-sample
  evaluation, fixed `q_bounds`, and bounded nuisance predictions that no bound reaches.
- Every fit declares `Runtime(random_state=0)`, and an unstratified fold draw depends only on the
  row count and the seed. Every replication of one size therefore uses the same assignment of rows
  to ten equal folds. The rows are iid, so that fixed assignment has the distribution of a random
  partition of each sample. The study does not measure how the estimate varies across fold draws
  of one sample.
- The primary and calibration rows use depth-five trees that can fit the saturated, correct model.
  Those rows do not test calibration under nuisance misspecification. The overfitting pairs are the
  data-adaptive evidence, and they use the exact mechanisms. The study does not establish
  performance for every learner library or tuning procedure.
- The exact efficient standard errors describe the finite laws. No calibration cell claims
  efficiency-bound attainment.
- The band cells measure the default Rademacher multiplier band over one law's reported estimands.
  They do not cover other multiplier kinds, other estimand sets, or bands across laws.
- The external comparison conditions on the stitched nuisance predictions from `cleverly`. It does
  not compare nuisance training. The continuous comparison rests on the planted-row scale
  workaround, which the probe checks on replication 0 only.
- The study excludes repeated splits, stratified or supplied folds, fold targeting or evaluation,
  `att` and `atc`, continuous `rr` and `or`, weights, clusters, baseline strata, bootstrap
  intervals, intermediates, missing treatment, MNAR outcomes, and C-TMLE or DR-TMLE fits.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_arm_indexed_cvtmle/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_arm_indexed_cvtmle/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_arm_indexed_cvtmle/replicates.csv.gz),
[scale probe](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_arm_indexed_cvtmle/scale-probe.csv),
[performance decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_arm_indexed_cvtmle/performance-tests.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_arm_indexed_cvtmle/properties.csv)
carry the protocol, provenance, and every published row.
