# Projection evidence (CDE, pooled regression, band)

This study validates three parts of the projection TMLE of a conditional contrast that
[conditional-projection](conditional-projection.md) left unmeasured. The first is the projection of
a controlled direct effect, `ControlledDirectEffect(intermediate=z,
contrast=ConditionalContrast(..., model=basis))`, at each level of a binary intermediate. The
second is one pooled outcome regression in place of one per arm. The third is the simultaneous band
of `effect_curve(grid=)`. The target is the coefficient vector of the least-squares projection of
the CATE, or of the controlled direct effect with $Z$ held at $z$, on a fixed basis, on the
difference scale, and of the log conditional risk ratio on the `log_ratio` scale.
[Conditional effects](../conditional-effects.md) gives the targets, the estimator and its
conditions.

**No canonical implementation is compared.** No maintained package fits a projection TMLE of a
controlled direct effect, or a simultaneous band over a projection TMLE's curve. A zero-row
equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | `cde_obs`: conditional-projection's observational law with a binary intermediate between the treatment and the outcome, $P(Z = 1 \mid A, W) = \operatorname{expit}(-0.2 + 0.8 A + 1.2 W_3 - 0.4 W_1)$ and $\operatorname{logit} \bar Q(a, z, W) = -0.3 + 0.6 W_1 - 0.4 W_2 + a\,b(W) + z (0.5 - 0.9 a)$ with the shipped blip $b(W) = 0.3 + 0.5 W_1 + 0.4 \sin(\pi W_3)$; conditional-projection's `observational` and `clustered` (800 clusters of 5 rows) laws |
| bases | `Linear(["W1"])` on $W_1$ on `cde_obs`, where $Z$ depends on $W_3$, which drives the nonlinear part of the blip; `Linear(["W1", "W3"])` on the other laws |
| groups | `cde_4k`, `cde_crr_4k` (the CDE projection at both levels from one cross-fitted fit with one outcome regression per arm, five outer folds), `cde_dr_4k` (a declared wrong outcome regression with the true mechanisms), `cde_total_4k` (the total-effect projection, a control), `pooled_4k`, `pooled_crr_4k` (one pooled outcome regression), and `band_obs_4k`, `band_crr_4k`, `band_clustered_4k`, `band_rep3_4k` (the band on the keys of rows `T49`, `T50`, `T49.clustered` and `T49.repeats`) |
| learners | a spline GLM per arm for $\mu$ with the raw $Z$ on `cde_obs`; a pooled spline GLM in $W$ with $A$ and $A \times$ spline$(W)$ for the pooled groups; conditional-projection's spline GLM per arm for the band groups; logistic GLMs for the treatment and intermediate mechanisms |
| band | the shipped band: Rademacher multipliers, 1,000 draws, the identity scale and the normal reference, over a $3 \times 3$ grid of $(W_1, W_3) \in \{-0.8, 0, 0.8\}^2$, and over five points of $W_1$ in $[-0.8, 0.8]$ on `cde_obs` |
| truth | Gauss-Legendre quadrature over $U(-1, 1)^3$, checked in the fast tier |
| primary budget | 2,400 replications of `cde_4k` at n = 4,000, the intercepts at both levels |
| calibration budget | R = 12,000 per calibration group, the plan's budget, which the rule computed before the run from a 3,200-draw pilot on a disjoint stream met (`tests/studies/projection_evidence_pilot.csv`); R = 2,400 per band group, 1,000 for `cde_dr_4k` and 500 for `cde_total_4k` |
| primary verdicts | the framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20261511` and `resampling_seed=20261512`; the run, the pilot and the smoke each draw from their own labelled streams, with fold seeds distinct per group |
| policy | `gated` |

The laws, groups, learners, budgets, controls and seeds were fixed before any run, in the P1 PR 2
plan, its reviews and `tests/studies/projection_evidence.py`.
`tests/unit/test_projection_evidence_design.py` checks the truths, the control limits, the
declared cells, the pilot rule and the budget.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, a spline GLM per arm with the raw Z and logistic treatment and intermediate models, five outer folds, n = 4,000 | `z0:cate_projection[(intercept)]` | the intercept of the least-squares projection of the CDE with Z held at 0 on (1, W1) | `cleverly` projection TMLE of a conditional contrast: the coefficients of the least-squares projection of the CATE on a fixed basis, targeted jointly, with an influence-curve Wald interval | -0.0013 to 0.0014 | 0.9508 | 1.0071 | pass |
| the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, a spline GLM per arm with the raw Z and logistic treatment and intermediate models, five outer folds, n = 4,000 | `z1:cate_projection[(intercept)]` | the intercept of the least-squares projection of the CDE with Z held at 1 on (1, W1) | `cleverly` projection TMLE of a conditional contrast: the coefficients of the least-squares projection of the CATE on a fixed basis, targeted jointly, with an influence-curve Wald interval | -0.000727 to 0.0016 | 0.9587 | 1.0318 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | for each coefficient of the CDE projection at each level on both scales and of the pooled projection on both scales (R = 12,000), does the SE-ratio interval lie inside 0.93 to 1.07 and the coverage interval inside 0.92 to 0.98 | the reported SE of the `cde_4k` intercept at $z = 0$ times 0.70 must fall below the band |
| `bias` | at a declared wrong outcome regression with the true mechanisms (R = 1,000), is each CDE coefficient's standardized-bias interval inside $\pm 0.25$ | each control's interval must lie wholly outside $\pm 0.25$: the targeted contrast averaged over the $Z = z$ rows only, the TMLE without the selection $1\{Z = z\}$, the TMLE without the intermediate density $q_z$, the plug-in of the wrong regression, and the total-effect projection read against the CDE truth |
| `simultaneous_coverage` | on the four band keys (R = 2,400), does the joint coverage interval of the band over the grid lie inside 0.92 to 0.98 | the pointwise 95% intervals over the same grid, read jointly, must have a joint coverage whose upper end lies below 0.95 |

The bands of `cde_4k` at each level and of `pooled_4k` publish their joint coverage under the
diagnostic role and state no verdict, so they are not among the gated cells.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `bias` | `cde_dr_4k_z0_W1` | positive | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, with a declared wrong outcome regression and the true treatment and intermediate mechanisms, five outer folds, n = 4,000: the TMLE of the W1 slope of the projection of the CDE with Z held at 0 on (1, W1) | the 99% interval of the standardized bias inside +-0.25 | bias -0.0036 to 0.0037, margin 0.0111 | pass |
| `bias` | `cde_dr_4k_z0_intercept` | positive | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, with a declared wrong outcome regression and the true treatment and intermediate mechanisms, five outer folds, n = 4,000: the TMLE of the intercept of the projection of the CDE with Z held at 0 on (1, W1) | the 99% interval of the standardized bias inside +-0.25 | bias -0.0030 to 0.0013, margin 0.0064 | pass |
| `bias` | `cde_dr_4k_z1_W1` | positive | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, with a declared wrong outcome regression and the true treatment and intermediate mechanisms, five outer folds, n = 4,000: the TMLE of the W1 slope of the projection of the CDE with Z held at 1 on (1, W1) | the 99% interval of the standardized bias inside +-0.25 | bias -0.0041 to 0.0024, margin 0.0100 | pass |
| `bias` | `cde_dr_4k_z1_intercept` | positive | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, with a declared wrong outcome regression and the true treatment and intermediate mechanisms, five outer folds, n = 4,000: the TMLE of the intercept of the projection of the CDE with Z held at 1 on (1, W1) | the 99% interval of the standardized bias inside +-0.25 | bias -0.0015 to 0.0021, margin 0.0055 | pass |
| `bias` | `no_intermediate_weight_4k_z0_intercept` | control | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, with a declared wrong outcome regression and the true treatment and intermediate mechanisms, five outer folds, n = 4,000: the TMLE with the intermediate density q_z omitted from the clever covariate, the intercept of the projection of the CDE with Z held at 0 on (1, W1) | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias 0.0770 to 0.0807, margin 0.0058 | pass |
| `bias` | `no_intermediate_weight_4k_z1_intercept` | control | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, with a declared wrong outcome regression and the true treatment and intermediate mechanisms, five outer folds, n = 4,000: the TMLE with the intermediate density q_z omitted from the clever covariate, the intercept of the projection of the CDE with Z held at 1 on (1, W1) | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0700 to -0.0666, margin 0.0051 | pass |
| `bias` | `no_selection_4k_z0_intercept` | control | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, with a declared wrong outcome regression and the true treatment and intermediate mechanisms, five outer folds, n = 4,000: the TMLE with the selection 1{Z = z} dropped from the observed clever covariate, the intercept of the projection of the CDE with Z held at 0 on (1, W1) | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0952 to -0.0930, margin 0.0034 | pass |
| `bias` | `no_selection_4k_z1_intercept` | control | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, with a declared wrong outcome regression and the true treatment and intermediate mechanisms, five outer folds, n = 4,000: the TMLE with the selection 1{Z = z} dropped from the observed clever covariate, the intercept of the projection of the CDE with Z held at 1 on (1, W1) | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias 0.0658 to 0.0681, margin 0.0035 | pass |
| `bias` | `plugin_stratum_4k_z0_intercept` | control | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, a spline GLM per arm with the raw Z and logistic treatment and intermediate models, five outer folds, n = 4,000: the targeted contrast projected over the Z = z rows only, read against the CDE truth, the intercept of the projection of the CDE with Z held at 0 on (1, W1) | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0181 to -0.0170, margin 0.0060 | pass |
| `bias` | `plugin_stratum_4k_z1_intercept` | control | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, a spline GLM per arm with the raw Z and logistic treatment and intermediate models, five outer folds, n = 4,000: the targeted contrast projected over the Z = z rows only, read against the CDE truth, the intercept of the projection of the CDE with Z held at 1 on (1, W1) | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias 0.0141 to 0.0152, margin 0.0054 | pass |
| `bias` | `plugin_wrong_4k_z0_W1` | control | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, with a declared wrong outcome regression and the true treatment and intermediate mechanisms, five outer folds, n = 4,000: the plug-in of the declared wrong outcome regression, with no targeting, the W1 slope of the projection of the CDE with Z held at 0 on (1, W1) | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0468 to -0.0455, margin 0.0020 | pass |
| `bias` | `plugin_wrong_4k_z1_W1` | control | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, with a declared wrong outcome regression and the true treatment and intermediate mechanisms, five outer folds, n = 4,000: the plug-in of the declared wrong outcome regression, with no targeting, the W1 slope of the projection of the CDE with Z held at 1 on (1, W1) | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0419 to -0.0406, margin 0.0019 | pass |
| `bias` | `total_effect_4k_z0_intercept` | control | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, fitted without the intermediate, n = 4,000: the total-effect projection, fitted without the intermediate, read against the CDE truth, the intercept of the projection of the CDE with Z held at 0 on (1, W1) | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.1117 to -0.1079, margin 0.0041 | pass |
| `bias` | `total_effect_4k_z1_intercept` | control | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, fitted without the intermediate, n = 4,000: the total-effect projection, fitted without the intermediate, read against the CDE truth, the intercept of the projection of the CDE with Z held at 1 on (1, W1) | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias 0.0956 to 0.0994, margin 0.0041 | pass |
| `interval_calibration` | `cde_4k_z0_W1__cde_z0` | positive | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, a spline GLM per arm with the raw Z and logistic treatment and intermediate models, five outer folds, n = 4,000: the influence-curve Wald interval of the W1 slope of the projection of the CDE with Z held at 0 on (1, W1) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9458 to 0.9560, SE ratio 0.9887 to 1.0221 | pass |
| `interval_calibration` | `cde_4k_z0_intercept__cde_z0` | positive | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, a spline GLM per arm with the raw Z and logistic treatment and intermediate models, five outer folds, n = 4,000: the influence-curve Wald interval of the intercept of the projection of the CDE with Z held at 0 on (1, W1) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9463 to 0.9565, SE ratio 0.9869 to 1.0204 | pass |
| `interval_calibration` | `cde_4k_z0_intercept__shrunken_se_control` | control | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, a spline GLM per arm with the raw Z and logistic treatment and intermediate models, five outer folds, n = 4,000: the reported SE of the intercept of the projection of the CDE with Z held at 0 on (1, W1) times 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8195 to 0.8373, SE ratio 0.6911 to 0.7143 | pass |
| `interval_calibration` | `cde_4k_z1_W1__cde_z1` | positive | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, a spline GLM per arm with the raw Z and logistic treatment and intermediate models, five outer folds, n = 4,000: the influence-curve Wald interval of the W1 slope of the projection of the CDE with Z held at 1 on (1, W1) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9437 to 0.9541, SE ratio 0.9878 to 1.0211 | pass |
| `interval_calibration` | `cde_4k_z1_intercept__cde_z1` | positive | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, a spline GLM per arm with the raw Z and logistic treatment and intermediate models, five outer folds, n = 4,000: the influence-curve Wald interval of the intercept of the projection of the CDE with Z held at 1 on (1, W1) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9437 to 0.9541, SE ratio 0.9899 to 1.0226 | pass |
| `interval_calibration` | `cde_crr_4k_z0_W1__cde_crr_z0` | positive | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, on the log-ratio scale, a spline GLM per arm with the raw Z, five outer folds, n = 4,000: the influence-curve Wald interval of the W1 slope of the projection of the log CRR with Z held at 0 on (1, W1) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9402 to 0.9509, SE ratio 0.9720 to 1.0053 | pass |
| `interval_calibration` | `cde_crr_4k_z0_intercept__cde_crr_z0` | positive | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, on the log-ratio scale, a spline GLM per arm with the raw Z, five outer folds, n = 4,000: the influence-curve Wald interval of the intercept of the projection of the log CRR with Z held at 0 on (1, W1) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9419 to 0.9524, SE ratio 0.9718 to 1.0053 | pass |
| `interval_calibration` | `cde_crr_4k_z1_W1__cde_crr_z1` | positive | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, on the log-ratio scale, a spline GLM per arm with the raw Z, five outer folds, n = 4,000: the influence-curve Wald interval of the W1 slope of the projection of the log CRR with Z held at 1 on (1, W1) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9451 to 0.9554, SE ratio 0.9789 to 1.0119 | pass |
| `interval_calibration` | `cde_crr_4k_z1_intercept__cde_crr_z1` | positive | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, on the log-ratio scale, a spline GLM per arm with the raw Z, five outer folds, n = 4,000: the influence-curve Wald interval of the intercept of the projection of the log CRR with Z held at 1 on (1, W1) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9495 to 0.9594, SE ratio 0.9916 to 1.0256 | pass |
| `interval_calibration` | `pooled_4k_W1__pooled` | positive | the observational law, one pooled outcome regression (a spline GLM in W with A and A x spline(W)), n = 4,000: the influence-curve Wald interval of the W1 slope of the projection of the CATE on (1, W1, W3) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9426 to 0.9532, SE ratio 0.9790 to 1.0123 | pass |
| `interval_calibration` | `pooled_4k_W3__pooled` | positive | the observational law, one pooled outcome regression (a spline GLM in W with A and A x spline(W)), n = 4,000: the influence-curve Wald interval of the W3 slope of the projection of the CATE on (1, W1, W3) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9426 to 0.9532, SE ratio 0.9778 to 1.0111 | pass |
| `interval_calibration` | `pooled_4k_intercept__pooled` | positive | the observational law, one pooled outcome regression (a spline GLM in W with A and A x spline(W)), n = 4,000: the influence-curve Wald interval of the intercept of the projection of the CATE on (1, W1, W3) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9415 to 0.9521, SE ratio 0.9713 to 1.0039 | pass |
| `interval_calibration` | `pooled_crr_4k_W1__pooled_crr` | positive | the observational law on the log-ratio scale, one pooled outcome regression (a spline GLM in W with A and A x spline(W)), n = 4,000: the influence-curve Wald interval of the W1 slope of the projection of the log CRR on (1, W1, W3) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9435 to 0.9540, SE ratio 0.9763 to 1.0092 | pass |
| `interval_calibration` | `pooled_crr_4k_W3__pooled_crr` | positive | the observational law on the log-ratio scale, one pooled outcome regression (a spline GLM in W with A and A x spline(W)), n = 4,000: the influence-curve Wald interval of the W3 slope of the projection of the log CRR on (1, W1, W3) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9441 to 0.9545, SE ratio 0.9776 to 1.0109 | pass |
| `interval_calibration` | `pooled_crr_4k_intercept__pooled_crr` | positive | the observational law on the log-ratio scale, one pooled outcome regression (a spline GLM in W with A and A x spline(W)), n = 4,000: the influence-curve Wald interval of the intercept of the projection of the log CRR on (1, W1, W3) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9431 to 0.9536, SE ratio 0.9719 to 1.0055 | pass |
| `simultaneous_coverage` | `band_clustered_4k__pointwise_joint_control` | control | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level, a spline GLM per arm, n = 4,000: the pointwise 95% intervals over a 3 x 3 grid of (W1, W3) in {-0.8, 0, 0.8}, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.7420 to 0.7870 | pass |
| `simultaneous_coverage` | `band_clustered_4k__simultaneous_band` | positive | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level, a spline GLM per arm, n = 4,000: the Rademacher multiplier band of effect_curve(grid=) with 1,000 draws, on the identity scale with the normal reference, over a 3 x 3 grid of (W1, W3) in {-0.8, 0, 0.8} | joint coverage interval inside the calibration coverage band | joint coverage 0.9346 to 0.9585 | pass |
| `simultaneous_coverage` | `band_crr_4k__pointwise_joint_control` | control | the observational law on the log-ratio scale, a spline GLM per arm, n = 4,000: the pointwise 95% intervals over a 3 x 3 grid of (W1, W3) in {-0.8, 0, 0.8}, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.7583 to 0.8022 | pass |
| `simultaneous_coverage` | `band_crr_4k__simultaneous_band` | positive | the observational law on the log-ratio scale, a spline GLM per arm, n = 4,000: the Rademacher multiplier band of effect_curve(grid=) with 1,000 draws, on the identity scale with the normal reference, over a 3 x 3 grid of (W1, W3) in {-0.8, 0, 0.8} | joint coverage interval inside the calibration coverage band | joint coverage 0.9333 to 0.9574 | pass |
| `simultaneous_coverage` | `band_obs_4k__pointwise_joint_control` | control | the observational law, a spline GLM per arm, n = 4,000: the pointwise 95% intervals over a 3 x 3 grid of (W1, W3) in {-0.8, 0, 0.8}, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.7424 to 0.7874 | pass |
| `simultaneous_coverage` | `band_obs_4k__simultaneous_band` | positive | the observational law, a spline GLM per arm, n = 4,000: the Rademacher multiplier band of effect_curve(grid=) with 1,000 draws, on the identity scale with the normal reference, over a 3 x 3 grid of (W1, W3) in {-0.8, 0, 0.8} | joint coverage interval inside the calibration coverage band | joint coverage 0.9328 to 0.9570 | pass |
| `simultaneous_coverage` | `band_rep3_4k__pointwise_joint_control` | control | the observational law, a spline GLM per arm with repeats=3 and the median rule, n = 4,000: the pointwise 95% intervals over a 3 x 3 grid of (W1, W3) in {-0.8, 0, 0.8}, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.7678 to 0.8110 | pass |
| `simultaneous_coverage` | `band_rep3_4k__simultaneous_band` | positive | the observational law, a spline GLM per arm with repeats=3 and the median rule, n = 4,000: the Rademacher multiplier band of effect_curve(grid=) with 1,000 draws, on the identity scale with the normal reference, over a 3 x 3 grid of (W1, W3) in {-0.8, 0, 0.8} | joint coverage interval inside the calibration coverage band | joint coverage 0.9356 to 0.9593 | pass |
| `simultaneous_coverage` | `cde_4k_z0__simultaneous_band` | diagnostic | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, a spline GLM per arm with the raw Z and logistic treatment and intermediate models, five outer folds, n = 4,000: the Rademacher multiplier band of effect_curve(grid=) with 1,000 draws, on the identity scale with the normal reference, over five points W1 in [-0.8, 0.8], with Z held at 0 | none. The row is a diagnostic and states no verdict | joint coverage 0.9422 to 0.9528 | reported |
| `simultaneous_coverage` | `cde_4k_z1__simultaneous_band` | diagnostic | the cde_obs law, a binary intermediate Z between the treatment and the binary outcome, with P(Z = 1) = expit(-0.2 + 0.8 A + 1.2 W3 - 0.4 W1) given A and W, a spline GLM per arm with the raw Z and logistic treatment and intermediate models, five outer folds, n = 4,000: the Rademacher multiplier band of effect_curve(grid=) with 1,000 draws, on the identity scale with the normal reference, over five points W1 in [-0.8, 0.8], with Z held at 1 | none. The row is a diagnostic and states no verdict | joint coverage 0.9430 to 0.9535 | reported |
| `simultaneous_coverage` | `pooled_4k__simultaneous_band` | diagnostic | the observational law, one pooled outcome regression (a spline GLM in W with A and A x spline(W)), n = 4,000: the Rademacher multiplier band of effect_curve(grid=) with 1,000 draws, on the identity scale with the normal reference, over a 3 x 3 grid of (W1, W3) in {-0.8, 0, 0.8} | none. The row is a diagnostic and states no verdict | joint coverage 0.9433 to 0.9538 | reported |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed, and no fit failed. Each of the 14
calibration positives has an SE-ratio interval inside 0.93 to 1.07 and a coverage interval inside
0.92 to 0.98: the CDE projection at both levels on both scales, and the pooled projection on both
scales. The CDE projection under the wrong outcome regression is unbiased at both levels, and
each control fails as declared. The band covers jointly on all four keys, where the pointwise
intervals do not.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 2 of 2 | `performance-tests.csv` |
| gated property cells passed | 37 of 37 | `properties.csv` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |

The calibration SE ratios run from 0.9871 to 1.0086. The shrunken control reads an SE ratio near
0.70. Under the wrong outcome regression the largest standardized bias is -0.033. The controls
are displaced by -0.73 and 0.67 SD when the contrast is averaged over the $Z = z$ rows, by -6.92
and 4.76 SD without the selection, by 3.41 and -3.34 SD without $q_z$, by -5.86 and -5.29 SD for
the plug-in of the wrong regression, and by -6.76 and 6.00 SD for the total effect. The four
bands cover 0.9458 to 0.9483 jointly, against 0.765 to 0.790 for the pointwise intervals. The
diagnostic bands cover 0.9477 and 0.9484 for the CDE at the two levels and 0.9487 for the pooled
projection.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications |
| `n` | 4000 | observations per replication |
| `independent_tests_total` | 2 | parameter tests against the truth |
| `independent_tests_passed` | 2 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 37 | gated repeated-sampling property cells |
| `property_cells_passed` | 37 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0193 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9508 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9383 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9706 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0714 | highest bootstrap primary SE-ratio endpoint |
| `properties[interval_calibration/pooled_4k_intercept__pooled]:se_ratio` | 0.9871 | lowest calibration SE ratio |
| `properties[interval_calibration/cde_crr_4k_z1_intercept__cde_crr_z1]:se_ratio` | 1.0086 | highest calibration SE ratio |
| `properties[interval_calibration/cde_4k_z0_intercept__shrunken_se_control]:se_ratio` | 0.7024 | shrunken control |
| `properties[bias/cde_dr_4k_z0_intercept]:standardized_bias` | -0.0330 | CDE projection under a wrong outcome regression, largest |
| `properties[bias/plugin_stratum_4k_z0_intercept]:standardized_bias` | -0.7263 | contrast averaged over the $Z = 0$ rows |
| `properties[bias/plugin_stratum_4k_z1_intercept]:standardized_bias` | 0.6728 | contrast averaged over the $Z = 1$ rows |
| `properties[bias/no_selection_4k_z0_intercept]:standardized_bias` | -6.9224 | TMLE without the selection, $z = 0$ |
| `properties[bias/no_selection_4k_z1_intercept]:standardized_bias` | 4.7591 | TMLE without the selection, $z = 1$ |
| `properties[bias/no_intermediate_weight_4k_z0_intercept]:standardized_bias` | 3.4115 | TMLE without $q_z$, $z = 0$ |
| `properties[bias/no_intermediate_weight_4k_z1_intercept]:standardized_bias` | -3.3430 | TMLE without $q_z$, $z = 1$ |
| `properties[bias/plugin_wrong_4k_z0_W1]:standardized_bias` | -5.8622 | plug-in of the wrong regression, $z = 0$ |
| `properties[bias/plugin_wrong_4k_z1_W1]:standardized_bias` | -5.2886 | plug-in of the wrong regression, $z = 1$ |
| `properties[bias/total_effect_4k_z0_intercept]:standardized_bias` | -6.7590 | total effect against the CDE truth, $z = 0$ |
| `properties[bias/total_effect_4k_z1_intercept]:standardized_bias` | 6.0021 | total effect against the CDE truth, $z = 1$ |
| `properties[simultaneous_coverage/band_obs_4k__simultaneous_band]:coverage` | 0.9458 | band, `T49` key |
| `properties[simultaneous_coverage/band_crr_4k__simultaneous_band]:coverage` | 0.9463 | band, `T50` key |
| `properties[simultaneous_coverage/band_clustered_4k__simultaneous_band]:coverage` | 0.9475 | band, `T49.clustered` key |
| `properties[simultaneous_coverage/band_rep3_4k__simultaneous_band]:coverage` | 0.9483 | band, `T49.repeats` key |
| `properties[simultaneous_coverage/band_obs_4k__pointwise_joint_control]:coverage` | 0.7654 | pointwise intervals read jointly, lowest |
| `properties[simultaneous_coverage/band_rep3_4k__pointwise_joint_control]:coverage` | 0.7900 | pointwise intervals read jointly, highest |
| `properties[simultaneous_coverage/cde_4k_z0__simultaneous_band]:coverage` | 0.9477 | diagnostic CDE band, $z = 0$ |
| `properties[simultaneous_coverage/cde_4k_z1__simultaneous_band]:coverage` | 0.9484 | diagnostic CDE band, $z = 1$ |
| `properties[simultaneous_coverage/pooled_4k__simultaneous_band]:coverage` | 0.9487 | diagnostic pooled band |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the primary coverage lower endpoint must clear |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio sanity screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio sanity screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | factor of the shrunken control |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 24 | failed primary fits the bound allows |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope band, upper limit |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained package fits a projection TMLE of a controlled direct effect or a band over a projection TMLE's curve. The rows rest on accuracy against the truth and on the theory properties |
| The interval covers the projection | The interval covers the coefficients of the least-squares projection of the CATE, or of the controlled direct effect with $Z$ held at $z$, on the basis over the study's modifier distribution, not the effect where it lies outside the span of the basis |
| One composition of the CDE projection | The CDE projection was measured cross-fitted and stacked, with one outcome regression per arm, a binary intermediate, complete outcomes and default targeting. In sample, a pooled outcome regression, weights, clusters, repeats, the bootstrap, a declared treatment law, `randomized=True` and fold-wise targeting read experimental; each is exact on a discrete law in the fast tier |
| Consistent learners | Every calibration group fits learners consistent for its law. A wrong outcome regression was measured for bias only, with the true treatment and intermediate mechanisms; a misspecified intermediate or treatment model is not measured |
| The pooled regression | One pooled spline GLM with an $A \times$ spline$(W)$ interaction, stacked and cross-fitted, on the observational law with complete outcomes. The pooled regression with weights, clusters, missing outcomes, three arms or the bootstrap reads experimental |
| The band | The shipped Rademacher band with 1,000 draws on the identity scale with the normal reference, over a nine-point grid on the four measured keys. Another multiplier, scale, reference or fewer draws reads experimental, and so does the band on any other key; the bands on the CDE and pooled keys were published as diagnostics |
| Laws and sizes | n = 4,000 on smooth laws, with a basis in one or two modifiers. Smaller n and rougher laws rest on the theory |
| Compositions outside the rows | Beside an intermediate, the EP learner (`method="ep"`), `strata=`, the held-out curve checks, the score assessment and the targeting curve refuse, and so do missing outcomes with cross-fitting and a missing treatment |

## Reproduction

The declared run is `python -m tests.canonical.projection_evidence.regenerate`, whose docstring
gives the run form and the red-cell reading. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/projection_evidence/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/projection_evidence/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/projection_evidence/replicates.parquet)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/projection_evidence/properties.csv)
carry every published row.
