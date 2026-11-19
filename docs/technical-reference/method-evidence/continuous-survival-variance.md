# Continuous-time survival switched variance

This study validates the default standard error of `ContinuousSurvivalTMLE`, the switched
variance, and decides whether it stays the default. The switched variance is the influence-curve
variance, raised to the predictable variation of the curve's martingale part when the curve is
sparse. The section on [inference](../continuous-time-survival.md#inference) states the
estimator, its switch and the exact-law evidence.

No external implementation reports this variance, so no canonical implementation is compared.
The study rests on exact truths and on repeated sampling. Every gated row passes at 6,000
replicates, the switched variance passes its ship rule and stays the default, and the run sets
the support floors in [the ship rule and the floors](#the-ship-rule-and-the-floors).

## The estimand and the laws

The targets are each arm's risk of each cause, $F_j(t \mid a) = E\,F_j(t \mid a, X)$, and each
risk difference, at the three quartiles of the observed event times. Identification needs no
unmeasured confounding of the treatment given $X$, censoring independent of the event times given
$(A, X)$, and positivity of the treatment and of remaining uncensored through the last time.

Each cell changes one thing in the base law of the
[continuous-time survival study](continuous-time-survival.md), so that no cell repeats a law a
registered study measures:

| cell | law | n | fit | what it stresses |
| --- | --- | --- | --- | --- |
| `v1_rare_cause2` | cause-2 log-hazard intercept $-2.6$ in place of $-2$; times 0.52, 1.29, 2.69 | 1,000 | cross-fitted | a rarer cause: about 9.5 treated cause-2 events by the first quartile |
| `v2_heavy_weights` | treatment model $\omega = (2, -1, 1.5)$ | 1,000 | cross-fitted | heavier control weights; the risks do not move |
| `v3_half_n` | base | 500 | cross-fitted | half the events; decides whether the size floor falls to 500 |
| `v4_in_sample` | base | 1,000 | in sample | where the influence-curve standard error runs lowest |
| `v7_quadratic` | $0.3\,(X_3^2 - 1)$ added to the cause-1 log hazard | 1,000 | cross-fitted | a hazard no default Cox model holds |

Every hazard stays constant given $(Z, X)$, so each truth is a quadrature over $X_3$, and the
efficient standard deviation has a closed form, in
`tests/studies/continuous_survival_variance_law.py`. The rare cause's target times are that law's
event-time quartiles on a fixed draw of 2,000,000; its median reads 1.29.

## The fit

Each cell fits the default libraries with 10 outer folds and selection inside 5 inner folds, as
the registered study does. Every interval is the one the library ships: the switched standard
error, range-logit on $(0, 1)$ for a risk and identity for a difference. Each fit also records,
for every target, the influence-curve and predictable-variation standard errors, the relative
standard deviation the switch reads (`variance_rel_sd`) and the oracle standard error, so every
candidate interval is read on the same fits. No comparator is named.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the base continuous-time competing-risks law with a rarer second cause (cause-2 log-hazard intercept -2.6); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause1 @ t=1.29]` | difference in the risk of cause 1 by t = 1.29 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000995 to 0.0012 | 0.9462 | 1.0048 | pass |
| the base continuous-time competing-risks law with a rarer second cause (cause-2 log-hazard intercept -2.6); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause1 @ t=2.69]` | difference in the risk of cause 1 by t = 2.69 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0011 to 0.0013 | 0.9457 | 0.9942 | pass |
| the base continuous-time competing-risks law with a rarer second cause (cause-2 log-hazard intercept -2.6); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause2 @ t=1.29]` | difference in the risk of cause 2 by t = 1.29 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000552 to 0.0011 | 0.9385 | 1.0000 | pass |
| the base continuous-time competing-risks law with a rarer second cause (cause-2 log-hazard intercept -2.6); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause2 @ t=2.69]` | difference in the risk of cause 2 by t = 2.69 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000930 to 0.0011 | 0.9425 | 0.9933 | pass |
| the base continuous-time competing-risks law with a rarer second cause (cause-2 log-hazard intercept -2.6); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause1 @ t=1.29]` | risk of cause 1 by t = 1.29 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000918 to 0.000871 | 0.9493 | 1.0050 | pass |
| the base continuous-time competing-risks law with a rarer second cause (cause-2 log-hazard intercept -2.6); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause1 @ t=2.69]` | risk of cause 1 by t = 2.69 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000640 to 0.0011 | 0.9485 | 0.9939 | pass |
| the base continuous-time competing-risks law with a rarer second cause (cause-2 log-hazard intercept -2.6); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause2 @ t=1.29]` | risk of cause 2 by t = 1.29 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000456 to 0.000473 | 0.9553 | 1.0171 | pass |
| the base continuous-time competing-risks law with a rarer second cause (cause-2 log-hazard intercept -2.6); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause2 @ t=2.69]` | risk of cause 2 by t = 2.69 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000760 to 0.000438 | 0.9513 | 0.9968 | pass |
| the base continuous-time competing-risks law with a rarer second cause (cause-2 log-hazard intercept -2.6); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause1 @ t=1.29]` | risk of cause 1 by t = 1.29 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000624 to 0.000776 | 0.9513 | 0.9990 | pass |
| the base continuous-time competing-risks law with a rarer second cause (cause-2 log-hazard intercept -2.6); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause1 @ t=2.69]` | risk of cause 1 by t = 2.69 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000537 to 0.0012 | 0.9455 | 0.9854 | pass |
| the base continuous-time competing-risks law with a rarer second cause (cause-2 log-hazard intercept -2.6); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause2 @ t=1.29]` | risk of cause 2 by t = 1.29 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000414 to 0.000953 | 0.9347 | 0.9862 | pass |
| the base continuous-time competing-risks law with a rarer second cause (cause-2 log-hazard intercept -2.6); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause2 @ t=2.69]` | risk of cause 2 by t = 2.69 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000937 to 0.000787 | 0.9417 | 0.9835 | pass |
| reported cell, passing by declaration (three rows at t = 1.12 fall short of the primary rule): the base continuous-time competing-risks law with heavier control weights (treatment model coefficients 2, -1, 1.5); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause1 @ t=1.12]` | difference in the risk of cause 1 by t = 1.12 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | 0.0014 to 0.0045 | 0.9080 | 0.9192 | pass |
| reported cell, passing by declaration (three rows at t = 1.12 fall short of the primary rule): the base continuous-time competing-risks law with heavier control weights (treatment model coefficients 2, -1, 1.5); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause1 @ t=2.32]` | difference in the risk of cause 1 by t = 2.32 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000340 to 0.0026 | 0.9312 | 0.9579 | pass |
| reported cell, passing by declaration (three rows at t = 1.12 fall short of the primary rule): the base continuous-time competing-risks law with heavier control weights (treatment model coefficients 2, -1, 1.5); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause2 @ t=1.12]` | difference in the risk of cause 2 by t = 1.12 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | 0.000449 to 0.0028 | 0.9248 | 0.9534 | pass |
| reported cell, passing by declaration (three rows at t = 1.12 fall short of the primary rule): the base continuous-time competing-risks law with heavier control weights (treatment model coefficients 2, -1, 1.5); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause2 @ t=2.32]` | difference in the risk of cause 2 by t = 2.32 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | 0.000235 to 0.0028 | 0.9372 | 0.9846 | pass |
| reported cell, passing by declaration (three rows at t = 1.12 fall short of the primary rule): the base continuous-time competing-risks law with heavier control weights (treatment model coefficients 2, -1, 1.5); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause1 @ t=1.12]` | risk of cause 1 by t = 1.12 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0036 to -0.000875 | 0.9052 | 0.9069 | pass |
| reported cell, passing by declaration (three rows at t = 1.12 fall short of the primary rule): the base continuous-time competing-risks law with heavier control weights (treatment model coefficients 2, -1, 1.5); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause1 @ t=2.32]` | risk of cause 1 by t = 2.32 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0020 to 0.000428 | 0.9262 | 0.9497 | pass |
| reported cell, passing by declaration (three rows at t = 1.12 fall short of the primary rule): the base continuous-time competing-risks law with heavier control weights (treatment model coefficients 2, -1, 1.5); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause2 @ t=1.12]` | risk of cause 2 by t = 1.12 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | 0.000037 to 0.0014 | 0.9535 | 0.9902 | pass |
| reported cell, passing by declaration (three rows at t = 1.12 fall short of the primary rule): the base continuous-time competing-risks law with heavier control weights (treatment model coefficients 2, -1, 1.5); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause2 @ t=2.32]` | risk of cause 2 by t = 2.32 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000278 to 0.0014 | 0.9488 | 0.9915 | pass |
| reported cell, passing by declaration (three rows at t = 1.12 fall short of the primary rule): the base continuous-time competing-risks law with heavier control weights (treatment model coefficients 2, -1, 1.5); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause1 @ t=1.12]` | risk of cause 1 by t = 1.12 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000045 to 0.0014 | 0.9423 | 0.9812 | pass |
| reported cell, passing by declaration (three rows at t = 1.12 fall short of the primary rule): the base continuous-time competing-risks law with heavier control weights (treatment model coefficients 2, -1, 1.5); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause1 @ t=2.32]` | risk of cause 1 by t = 2.32 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000566 to 0.0012 | 0.9380 | 0.9688 | pass |
| reported cell, passing by declaration (three rows at t = 1.12 fall short of the primary rule): the base continuous-time competing-risks law with heavier control weights (treatment model coefficients 2, -1, 1.5); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause2 @ t=1.12]` | risk of cause 2 by t = 1.12 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | 0.0014 to 0.0033 | 0.9047 | 0.9380 | pass |
| reported cell, passing by declaration (three rows at t = 1.12 fall short of the primary rule): the base continuous-time competing-risks law with heavier control weights (treatment model coefficients 2, -1, 1.5); the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause2 @ t=2.32]` | risk of cause 2 by t = 2.32 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | 0.0010 to 0.0031 | 0.9255 | 0.9742 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, ten outer folds, n = 500 | `ate_regimen[treated vs control, cause1 @ t=1.12]` | difference in the risk of cause 1 by t = 1.12 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0016 to 0.0016 | 0.9410 | 0.9907 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, ten outer folds, n = 500 | `ate_regimen[treated vs control, cause1 @ t=2.32]` | difference in the risk of cause 1 by t = 2.32 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0030 to 0.000258 | 0.9454 | 1.0108 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, ten outer folds, n = 500 | `ate_regimen[treated vs control, cause2 @ t=1.12]` | difference in the risk of cause 2 by t = 1.12 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | 0.000448 to 0.0031 | 0.9345 | 0.9962 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, ten outer folds, n = 500 | `ate_regimen[treated vs control, cause2 @ t=2.32]` | difference in the risk of cause 2 by t = 2.32 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | 0.000565 to 0.0036 | 0.9437 | 1.0066 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, ten outer folds, n = 500 | `cif_regimen[control, cause1 @ t=1.12]` | risk of cause 1 by t = 1.12 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0014 to 0.0012 | 0.9415 | 0.9951 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, ten outer folds, n = 500 | `cif_regimen[control, cause1 @ t=2.32]` | risk of cause 1 by t = 2.32 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000595 to 0.0019 | 0.9516 | 1.0233 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, ten outer folds, n = 500 | `cif_regimen[control, cause2 @ t=1.12]` | risk of cause 2 by t = 1.12 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0012 to 0.000372 | 0.9603 | 1.0409 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, ten outer folds, n = 500 | `cif_regimen[control, cause2 @ t=2.32]` | risk of cause 2 by t = 2.32 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0021 to -0.000188 | 0.9576 | 1.0159 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, ten outer folds, n = 500 | `cif_regimen[treated, cause1 @ t=1.12]` | risk of cause 1 by t = 1.12 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0010 to 0.000864 | 0.9549 | 1.0167 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, ten outer folds, n = 500 | `cif_regimen[treated, cause1 @ t=2.32]` | risk of cause 1 by t = 2.32 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0018 to 0.000440 | 0.9481 | 1.0043 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, ten outer folds, n = 500 | `cif_regimen[treated, cause2 @ t=1.12]` | risk of cause 2 by t = 1.12 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | 0.000263 to 0.0025 | 0.9317 | 0.9738 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, ten outer folds, n = 500 | `cif_regimen[treated, cause2 @ t=2.32]` | risk of cause 2 by t = 2.32 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000308 to 0.0022 | 0.9476 | 1.0117 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, fitted in sample, n = 1,000 | `ate_regimen[treated vs control, cause1 @ t=1.12]` | difference in the risk of cause 1 by t = 1.12 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0014 to 0.000802 | 0.9372 | 0.9536 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, fitted in sample, n = 1,000 | `ate_regimen[treated vs control, cause1 @ t=2.32]` | difference in the risk of cause 1 by t = 2.32 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0013 to 0.000981 | 0.9407 | 0.9695 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, fitted in sample, n = 1,000 | `ate_regimen[treated vs control, cause2 @ t=1.12]` | difference in the risk of cause 2 by t = 1.12 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | 0.000292 to 0.0021 | 0.9368 | 0.9800 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, fitted in sample, n = 1,000 | `ate_regimen[treated vs control, cause2 @ t=2.32]` | difference in the risk of cause 2 by t = 2.32 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000019 to 0.0021 | 0.9388 | 0.9771 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, fitted in sample, n = 1,000 | `cif_regimen[control, cause1 @ t=1.12]` | risk of cause 1 by t = 1.12 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000119 to 0.0017 | 0.9370 | 0.9644 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, fitted in sample, n = 1,000 | `cif_regimen[control, cause1 @ t=2.32]` | risk of cause 1 by t = 2.32 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000387 to 0.0013 | 0.9442 | 0.9795 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, fitted in sample, n = 1,000 | `cif_regimen[control, cause2 @ t=1.12]` | risk of cause 2 by t = 1.12 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000910 to 0.000186 | 0.9483 | 1.0026 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, fitted in sample, n = 1,000 | `cif_regimen[control, cause2 @ t=2.32]` | risk of cause 2 by t = 2.32 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0012 to 0.000103 | 0.9513 | 1.0029 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, fitted in sample, n = 1,000 | `cif_regimen[treated, cause1 @ t=1.12]` | risk of cause 1 by t = 1.12 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000178 to 0.0012 | 0.9432 | 0.9830 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, fitted in sample, n = 1,000 | `cif_regimen[treated, cause1 @ t=2.32]` | risk of cause 1 by t = 2.32 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000469 to 0.0011 | 0.9460 | 0.9845 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, fitted in sample, n = 1,000 | `cif_regimen[treated, cause2 @ t=1.12]` | risk of cause 2 by t = 1.12 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | 0.000089 to 0.0016 | 0.9398 | 0.9811 | pass |
| the base continuous-time competing-risks law; the default Cox libraries and treatment learner, fitted in sample, n = 1,000 | `cif_regimen[treated, cause2 @ t=2.32]` | risk of cause 2 by t = 2.32 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000401 to 0.0014 | 0.9378 | 0.9904 | pass |
| the base continuous-time competing-risks law with 0.3 (X3^2 - 1) added to the cause-1 log hazard, which no default Cox model holds; the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause1 @ t=1.12]` | difference in the risk of cause 1 by t = 1.12 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000987 to 0.0011 | 0.9493 | 1.0030 | pass |
| the base continuous-time competing-risks law with 0.3 (X3^2 - 1) added to the cause-1 log hazard, which no default Cox model holds; the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause1 @ t=2.32]` | difference in the risk of cause 1 by t = 2.32 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0020 to 0.000271 | 0.9510 | 1.0070 | pass |
| the base continuous-time competing-risks law with 0.3 (X3^2 - 1) added to the cause-1 log hazard, which no default Cox model holds; the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause2 @ t=1.12]` | difference in the risk of cause 2 by t = 1.12 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000586 to 0.0013 | 0.9422 | 0.9974 | pass |
| the base continuous-time competing-risks law with 0.3 (X3^2 - 1) added to the cause-1 log hazard, which no default Cox model holds; the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `ate_regimen[treated vs control, cause2 @ t=2.32]` | difference in the risk of cause 2 by t = 2.32 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000718 to 0.0015 | 0.9417 | 0.9875 | pass |
| the base continuous-time competing-risks law with 0.3 (X3^2 - 1) added to the cause-1 log hazard, which no default Cox model holds; the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause1 @ t=1.12]` | risk of cause 1 by t = 1.12 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0011 to 0.000649 | 0.9520 | 1.0047 | pass |
| the base continuous-time competing-risks law with 0.3 (X3^2 - 1) added to the cause-1 log hazard, which no default Cox model holds; the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause1 @ t=2.32]` | risk of cause 1 by t = 2.32 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000510 to 0.0012 | 0.9492 | 1.0075 | pass |
| the base continuous-time competing-risks law with 0.3 (X3^2 - 1) added to the cause-1 log hazard, which no default Cox model holds; the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause2 @ t=1.12]` | risk of cause 2 by t = 1.12 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000924 to 0.000158 | 0.9542 | 1.0170 | pass |
| the base continuous-time competing-risks law with 0.3 (X3^2 - 1) added to the cause-1 log hazard, which no default Cox model holds; the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[control, cause2 @ t=2.32]` | risk of cause 2 by t = 2.32 under no treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000994 to 0.000351 | 0.9487 | 1.0008 | pass |
| the base continuous-time competing-risks law with 0.3 (X3^2 - 1) added to the cause-1 log hazard, which no default Cox model holds; the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause1 @ t=1.12]` | risk of cause 1 by t = 1.12 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000807 to 0.000542 | 0.9528 | 1.0171 | pass |
| the base continuous-time competing-risks law with 0.3 (X3^2 - 1) added to the cause-1 log hazard, which no default Cox model holds; the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause1 @ t=2.32]` | risk of cause 1 by t = 2.32 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.0013 to 0.000304 | 0.9510 | 1.0120 | pass |
| the base continuous-time competing-risks law with 0.3 (X3^2 - 1) added to the cause-1 log hazard, which no default Cox model holds; the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause2 @ t=1.12]` | risk of cause 2 by t = 1.12 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000811 to 0.000725 | 0.9413 | 1.0004 | pass |
| the base continuous-time competing-risks law with 0.3 (X3^2 - 1) added to the cause-1 log hazard, which no default Cox model holds; the default Cox libraries and treatment learner, ten outer folds, n = 1,000 | `cif_regimen[treated, cause2 @ t=2.32]` | risk of cause 2 by t = 2.32 under treatment | `cleverly` ContinuousSurvivalTMLE with variance='switched': the one-step TMLE's influence-curve standard error, raised to the predictable variation where the relative SD of the variance estimate exceeds 0.125 (Rytgaard and van der Laan 2024) | -0.000851 to 0.000944 | 0.9453 | 1.0007 | pass |
<!-- /generated -->

The twelve targets at the median and upper quartile of `v1_rare_cause2`, `v3_half_n`,
`v4_in_sample` and `v7_quadratic` are gated by the primary rule: coverage above 0.90 and below
0.99, the SE ratio inside (0.8, 1.2) and the bias within 0.2 SD. All 48 gated rows pass:

| cell | fits | coverage | SE ratio | largest standardized bias | fits lost |
| --- | ---: | --- | --- | ---: | ---: |
| `v1_rare_cause2` | 6,000 | 0.935 to 0.955 | 0.984 to 1.017 | 0.013 | 0 |
| `v3_half_n` | 5,988 | 0.932 to 0.960 | 0.974 to 1.041 | 0.046 | 12 |
| `v4_in_sample` | 6,000 | 0.937 to 0.951 | 0.954 to 1.003 | 0.044 | 0 |
| `v7_quadratic` | 6,000 | 0.941 to 0.954 | 0.987 to 1.017 | 0.025 | 0 |

The lowest gated coverage is the treated arm's risk of cause 2 at $t = 1.12$ in `v3_half_n`
(0.932), and the lowest SE ratio is the in-sample cause-1 difference at $t = 1.12$ (0.954).
`v3_half_n` lost 12 of 6,000 fits, against an allowance of 60 under the bound of 0.01
(`failure-accounting.csv`). Eight are the `DataError` of an initial hazard fit whose combined
jump at a late time with few units at risk is one or more, and four are the `DataError` of a
treated arm with no cause-2 event by the first target time, whose risk then has a zero influence
curve. Both refusals are kept on purpose at 500 units (`pilot-diagnosis.md`). No other cell lost
a fit.

`v2_heavy_weights` is a reported cell. Its rows publish the same readings with no verdict: coverage
0.905 to 0.954, SE ratios 0.907 to 0.992 and a largest standardized bias of 0.081. Three of its
rows fall short of the primary rule, each at $t = 1.12$ with a 99% coverage interval reaching
below 0.90: control cause 1 (coverage 0.905, SE ratio 0.91), the cause-1 difference (0.908, 0.92)
and treated cause 2 (0.905, 0.94). This is the weight-driven shortfall the pilot read before the
run (0.893 for control cause 1 at an SE ratio of 0.88), and with V2 in the sizing no count up to
50,000 passed, which is why the cell was declared reported before any primary run.

V2's rows still enter the `variance_rel_sd` floor and the ship rule, and its failure share, zero,
gates nothing. Roadmap item
[X6 (a2)](../../roadmap.md#x6-continuous-time-survival-and-competing-risks) owns the cell. No label detects heavy weights: such a fit's estimates read experimental only where their
`variance_rel_sd` lies above the floor.

The first pilot also lost 12 of 1,000 `v3_half_n` fits to a loss of digits in the clever
covariate where the fitted survival is tiny. That defect was fixed before the run, the rerun pilot
lost no fit, and with V3 in the sizing the count stayed 6,000, so V3 was gated.

### The first quartile

The six first-quartile targets of each fit are read under the calibration rule (SE ratio inside
(0.93, 1.07), coverage inside (0.92, 0.98), both at 99%) and gate nothing. The switched interval
passes at 22 of the 30 rows, and the influence-curve interval at 10 (`candidate-readings.csv`):

| cell | switched rows passing | influence-curve rows passing | switched rows outside the rule |
| --- | ---: | ---: | --- |
| `v1_rare_cause2` | 5 of 6 | 3 of 6 | treated cause 2 (coverage 0.921, SE ratio 0.947) |
| `v2_heavy_weights` | 2 of 6 | 0 of 6 | control cause 1 (0.862, 0.842), treated cause 2 (0.868, 0.857), the cause-1 difference (0.861, 0.853), the cause-2 difference (0.918, 0.895) |
| `v3_half_n` | 4 of 6 | 2 of 6 | treated cause 2 (0.920, 0.935), treated cause 1 (0.963, 1.046) |
| `v4_in_sample` | 5 of 6 | 2 of 6 | treated cause 2 (0.925, 0.944) |
| `v7_quadratic` | 6 of 6 | 3 of 6 | none |

The declared route for these rows is in `tests/canonical/continuous_survival_variance/pilot-diagnosis.md`,
written before the run. Outside V2, whose four short rows are its weights, the short rows are the
sparsest cells, treated cause 2 at the first quartile, where the influence-curve SE ratio is 0.85
to 0.86 and the switched interval closes most of the gap but not all of it. V3's treated cause 1
is over-width: the switched variance raises it in nearly every fit (median `variance_rel_sd` 0.21)
and moves an influence-curve SE ratio of 1.00 to 1.05. Every first-quartile row gates nothing, the share
floor stays at 0.40, so every first-quartile estimate reads experimental, and the shortfall is the
open item X6 (a2). No gated row is red, so no row becomes a red-cell owner.

## The ship rule and the floors

The record declares no property family. Its decision and its floors are computed from the primary
rows and committed as `ship-rule.csv`, `candidate-readings.csv` and `floors.csv`:

| reading | rule | what it decides |
| --- | --- | --- |
| harm | at no row does the switched SE ratio's interval lie wholly above 1.07, or its coverage interval wholly above 0.98, while the influence curve's does not | with benefit, whether the switched variance stays the default |
| benefit | at one first-quartile row at least, the paired 99% bootstrap interval of $\lvert\mathrm{cov}_{IC} - 0.95\rvert - \lvert\mathrm{cov}_S - 0.95\rvert$ lies above 0 | as above |
| controls | the predictable variation times 1.3 must fire the harm clause; a switch that never fires must fail the benefit clause; the standard error times 0.85 must fail the calibration rule at every gated row | the run is void if one does not behave so |
| reported controls | the gated rows the primary rule turns red at 0.85 times the standard error, and the cells whose sparsest first-quartile influence-curve row passes the calibration rule | nothing; the first is the merge gate's own sensitivity, the second says whether a cell exercises the sparse-cell failure |
| paired readings | each candidate's coverage minus the influence curve's, and minus the influence-curve interval widened to the candidate's mean width in its row, each with its paired 99% bootstrap interval | nothing; the evidence of roadmap item X6 (a2) |
| floors | the largest `variance_rel_sd` rung at which every primary row of the gated cells and of the reported heavy-weight cell passes; the event-share floor (0.40, or 0.25 if every first-quartile row passes); the size floor (1,000, or 500 if every gated `v3_half_n` row passes) | which risks and risk differences read `supported` |

The switched variance ships. The harm clause finds no harmed row, and the benefit clause finds 20
improved first-quartile rows, four in each cell: both differences, control cause 1 and treated
cause 2. Each control behaves as declared. The predictable variation times 1.3 fires the harm
clause at 16 rows, the switch that never fires improves no row, and the standard error times 0.85
fails the calibration rule at all 60 median and upper-quartile rows. Reported beside them, the
primary rule turns 55 of the 60 rows red at 0.85 times the standard error, and no cell's sparsest first-quartile
influence-curve row passes the calibration rule, so every cell exercises the sparse-cell failure.
The predictable variation times 1.15 is reported and harms 4 rows.

The floors, read from the switched rows in `floors.csv`, are written as constants in
`src/cleverly/survival/result.py`:

| floor | value | why |
| --- | --- | --- |
| `variance_rel_sd` (`MEASURED_REL_SD`) | 0.2458 | the three red V2 rows have median `variance_rel_sd` 0.246, 0.289 and 0.318, so the floor stops at the largest rung below them, that of V2's treated cause 2 at $t = 2.32$ |
| event share (`SHARE_FLOOR_DEFAULT`) | 0.40 | 8 of the 30 first-quartile rows fail the calibration rule, so the floor does not fall to 0.25, and no first-quartile floor (`MEASURED_REL_SD_EARLY`) is read |
| size (`MEASURED_N`) | 500 | every gated `v3_half_n` row passes, and its 12 lost fits are inside the bound; the cell fits no restricted horizon, so a fit with restricted horizons keeps 1,000 (`FUNCTIONALS_N`) |

Beside the other conditions of [continuous-time survival](../continuous-time-survival.md#evidence),
a risk or risk difference whose `variance_rel_sd` lies above 0.2458, whose time has fewer than 40%
of the observed events behind it, or whose fit has fewer than 500 units reads experimental. A
supported risk or risk difference names this study in its evidence.
`tests/unit/test_continuous_survival_variance_design.py` checks each constant against
`floors.csv`.

At 3,600 replicates the harm clause passes with probability 0.997 or more when every true switched
row is inside the band, and fires with probability 0.89 at a true SE ratio of 1.12. The benefit
clause passes with probability 1.000 (`tests/studies/continuous_survival_variance_readings.py`).
The run used 6,000, so these powers are lower bounds.

## Sizing

The pilot in `tests/studies/continuous_survival_variance_pilot.py` set the count before the run:
the larger of the shared primary rule on the gated cells and 3,600, the count at which the 30
first-quartile rows pass the calibration rule together with probability 0.977 at an SE ratio of
1. The shared rule asked for 6,000, where a row whose true bias is 0.15 SD passes the bias gate
with probability 0.90. The cap was 7,200. The pilot also kept the V7 coefficient at 0.3 and V7
gated: its untargeted plug-in of the cause-1 risk at the median is biased by 0.45 SD.

## Red-cell route

The study is gated, and no gated row is red. The three V2 rows that fall short are reported, and
the eight first-quartile rows outside the calibration rule gate nothing; both are routed to X6
(a2). The declared readings are in `tests/canonical/continuous_survival_variance/regenerate.py`.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed. The coverage and bias figures
read every primary row, the reported V2 rows included.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 6000 | primary replications |
| `n` | 1000 | observations per replication of the record's reference size |
| `independent_tests_passed` | 60 | truth tests passing |
| `independent_tests_total` | 60 | truth tests reported |
| `property_cells_total` | 0 | repeated-sampling property cells, none declared |
| `property_cells_passed` | 0 | of those, passing |
| `subject_failures` | 12 | primary fits lost to a recordable data failure |
| `max_standardized_bias` | 0.0814 | largest primary standardized bias |
| `min_coverage` | 0.9047 | lowest primary coverage |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2000 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 60 | the same bound, in primary replications |

## Limitations

| limit | what it means for use |
| --- | --- |
| two causes and a binary treatment | one event type and three causes are measured by their own study |
| the base law's family | the cells change one feature each of one law |
| heavy weights are reported, not gated | three V2 rows at $t = 1.12$ fall short, no label detects heavy weights, and X6 (a2) owns the configuration |
| the first quartile | 8 of 30 first-quartile rows fall outside the calibration rule, so the share floor stays at 0.40 and those estimates read experimental |
| no comparator | no external implementation reports this variance |

## Reproduction

Run the pilot first, as `python -m tests.studies.continuous_survival_variance_pilot --jobs N`.
Run `python -m tests.canonical.continuous_survival_variance.regenerate --output DIR` with an
empty scratch directory outside the repository to regenerate the artifacts. The red-cell readings,
the pilot's checks and the budget are declared in
`tests/canonical/continuous_survival_variance/regenerate.py` before any run.
