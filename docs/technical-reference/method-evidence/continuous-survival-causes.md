# Continuous-time survival with one and three causes

This study validates `ContinuousSurvivalTMLE` with a number of causes other than two: one event
type, where the risk is $1 - S(t)$, and three competing causes. The registered
[continuous-time survival study](continuous-time-survival.md) measures the same default fit with
two causes. [Continuous-time survival and competing risks](../continuous-time-survival.md) states
the estimator, its path and its stopping rule for any number of causes.

No maintained package implements this construction, so no canonical implementation is compared.
The study rests on exact truths and on repeated-sampling properties. Every gated row and every
property cell passes in the declared run at commit 820bdc39, so a fit with one cause or with three
like the one measured reads supported under the conditions of [limits](#limits).

## The estimand and the laws

For arm $a$ and cause $j$ the targets are the risk $F_j(t \mid a) = E\,F_j(t \mid a, X)$ and the
difference of the two arms. With one cause $F_1(t \mid a) = 1 - S(t \mid a)$, the risk of the
event. Identification needs no unmeasured confounding of the treatment given $X$, censoring
independent of the event times given $(A, X)$, and positivity of the treatment and of remaining
uncensored through $t$.

Both laws are the base law of the [continuous-time survival study](continuous-time-survival.md)
with the number of causes changed: the same covariates, the propensity
$\operatorname{expit}(X_1 - X_2 + X_3)$, the arm-specific term $Z X_4$ in $h_1$ that only the
interaction member of the default library holds, and exponential censoring at rate 0.15.

| law | causes | hazards given $(Z, X)$ | target times |
| --- | --- | --- | --- |
| L1 | one | the base law's $h_1$, with $h_2 \equiv 0$ | 0.58, 1.48, 3.24 |
| L3 | three | the base law's $h_1$ and $h_2$, and $h_3 = \exp(-2.5 + 0.5 X_1 + 0.5 X_3 + 0.5 Z)$ | 0.37, 0.9, 1.86 |

Every hazard is constant in time given $(Z, X)$, so $F_j(t \mid z, x) = (h_j / h)(1 - e^{-ht})$
with $h = \sum_l h_l$, and each truth is the base law's quadrature over $X_3$. The target times
are each law's event-time quartiles, read from a draw of 2,000,000. The laws and truths are in
`tests/studies/continuous_survival_causes_law.py`, and
`tests/unit/test_continuous_survival_causes_design.py` checks them against Monte Carlo.

## The fit

The fit is the registered study's gated configuration: the default hazard, censoring and
treatment libraries, cross-fitted with 10 outer folds and selection inside 5 inner folds, both
arms tilted, at n = 1,000, with the three target times fitted jointly. Every interval is the one
the library ships, with the default switched standard error
([the switched variance](../continuous-time-survival.md#the-switched-variance)), on the identity
scale. The gated times are each law's median and upper quartile: 6 rows for L1 and 18 for L3. The
switched-variance study kept the share floor at 0.40, so the first quartile is not licensed, and
its targets are published without a verdict in [the first quartile](#the-first-quartile).

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the base continuous-time law with its cause-2 hazard removed, so every event is of one type; the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.58, 1.48 and 3.24 | `ate_regimen[treated vs control @ t=1.48]` | difference in the risk of the event by t = 1.48 between treatment and no treatment, one cause | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.0012 to 0.000861 | 0.9517 | 1.0167 | pass |
| the base continuous-time law with its cause-2 hazard removed, so every event is of one type; the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.58, 1.48 and 3.24 | `ate_regimen[treated vs control @ t=3.24]` | difference in the risk of the event by t = 3.24 between treatment and no treatment, one cause | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000843 to 0.0013 | 0.9500 | 1.0139 | pass |
| the base continuous-time law with its cause-2 hazard removed, so every event is of one type; the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.58, 1.48 and 3.24 | `risk_regimen[control @ t=1.48]` | risk of the event by t = 1.48 under no treatment, one cause | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000547 to 0.0011 | 0.9503 | 1.0223 | pass |
| the base continuous-time law with its cause-2 hazard removed, so every event is of one type; the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.58, 1.48 and 3.24 | `risk_regimen[control @ t=3.24]` | risk of the event by t = 3.24 under no treatment, one cause | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000806 to 0.000705 | 0.9554 | 1.0239 | pass |
| the base continuous-time law with its cause-2 hazard removed, so every event is of one type; the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.58, 1.48 and 3.24 | `risk_regimen[treated @ t=1.48]` | risk of the event by t = 1.48 under treatment, one cause | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000558 to 0.000774 | 0.9539 | 1.0148 | pass |
| the base continuous-time law with its cause-2 hazard removed, so every event is of one type; the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.58, 1.48 and 3.24 | `risk_regimen[treated @ t=3.24]` | risk of the event by t = 3.24 under treatment, one cause | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000649 to 0.0010 | 0.9531 | 1.0185 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `ate_regimen[treated vs control, cause1 @ t=0.9]` | difference in the risk of cause 1 of three by t = 0.9 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000456 to 0.0014 | 0.9400 | 0.9973 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `ate_regimen[treated vs control, cause1 @ t=1.86]` | difference in the risk of cause 1 of three by t = 1.86 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000150 to 0.0019 | 0.9472 | 0.9980 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `ate_regimen[treated vs control, cause2 @ t=0.9]` | difference in the risk of cause 2 of three by t = 0.9 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000295 to 0.0013 | 0.9406 | 1.0022 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `ate_regimen[treated vs control, cause2 @ t=1.86]` | difference in the risk of cause 2 of three by t = 1.86 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000532 to 0.0013 | 0.9429 | 0.9955 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `ate_regimen[treated vs control, cause3 @ t=0.9]` | difference in the risk of cause 3 of three by t = 0.9 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.0010 to 0.000452 | 0.9553 | 1.0258 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `ate_regimen[treated vs control, cause3 @ t=1.86]` | difference in the risk of cause 3 of three by t = 1.86 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.0015 to 0.000289 | 0.9504 | 1.0163 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `cif_regimen[control, cause1 @ t=0.9]` | risk of cause 1 of three by t = 0.9 under no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.0013 to 0.000349 | 0.9422 | 0.9960 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `cif_regimen[control, cause1 @ t=1.86]` | risk of cause 1 of three by t = 1.86 under no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.0017 to 0.000013 | 0.9482 | 0.9990 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `cif_regimen[control, cause2 @ t=0.9]` | risk of cause 2 of three by t = 0.9 under no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000662 to 0.000244 | 0.9540 | 1.0238 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `cif_regimen[control, cause2 @ t=1.86]` | risk of cause 2 of three by t = 1.86 under no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000866 to 0.000259 | 0.9533 | 1.0133 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `cif_regimen[control, cause3 @ t=0.9]` | risk of cause 3 of three by t = 0.9 under no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000103 to 0.000976 | 0.9472 | 1.0188 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `cif_regimen[control, cause3 @ t=1.86]` | risk of cause 3 of three by t = 1.86 under no treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | 0.000082 to 0.0014 | 0.9522 | 1.0270 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `cif_regimen[treated, cause1 @ t=0.9]` | risk of cause 1 of three by t = 0.9 under treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000498 to 0.000566 | 0.9582 | 1.0251 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `cif_regimen[treated, cause1 @ t=1.86]` | risk of cause 1 of three by t = 1.86 under treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000599 to 0.000720 | 0.9526 | 1.0114 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `cif_regimen[treated, cause2 @ t=0.9]` | risk of cause 2 of three by t = 0.9 under treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000382 to 0.000942 | 0.9301 | 0.9798 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `cif_regimen[treated, cause2 @ t=1.86]` | risk of cause 2 of three by t = 1.86 under treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000678 to 0.000865 | 0.9469 | 1.0037 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `cif_regimen[treated, cause3 @ t=0.9]` | risk of cause 3 of three by t = 0.9 under treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000351 to 0.000663 | 0.9563 | 1.0212 | pass |
| the base continuous-time competing-risks law with a third cause of hazard exp(-2.5 + 0.5 X1 + 0.5 X3 + 0.5 Z); the default Cox libraries and treatment learner, ten outer folds, n = 1,000, target times 0.37, 0.9 and 1.86 | `cif_regimen[treated, cause3 @ t=1.86]` | risk of cause 3 of three by t = 1.86 under treatment | `cleverly` ContinuousSurvivalTMLE: one-step TMLE on the multinomial-logistic universal least favourable path of every cause hazard, both arms tilted, with the default switched standard error (Rytgaard and van der Laan 2024), with one cause and with three | -0.000521 to 0.000752 | 0.9550 | 1.0169 | pass |
<!-- /generated -->

Every primary row passes over 7,200 fits per law, and no fit failed. L1's six rows cover 0.950 to
0.955 with SE ratios 1.014 to 1.024. L3's 18 rows cover 0.930 to 0.958 with SE ratios 0.980 to
1.027. The lowest row is L3's treated risk of cause 2 at $t = 0.9$ (coverage 0.930, SE ratio
0.980), the sparsest cell of that law. The largest standardized bias is 0.034, on L3's control
risk of cause 3 at $t = 1.86$.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `l1_c1__factual_tilt_control` | control | the one-cause law, the treated arm's risk of the event by t = 1.48, ten outer folds, n = 1,000: the event hazards wrong (a Cox model on the treatment alone), with the default treatment learner and censoring library, which hold the true models; the same draw fitted with tilt='factual', which tilts only each unit's received arm | bias interval must fall entirely outside the margin | bias 0.0458 to 0.0485, margin 0.0041, SE ratio 1.1666 | pass |
| `double_robustness` | `l1_c1__mechanism_correct` | positive | the one-cause law, the treated arm's risk of the event by t = 1.48, ten outer folds, n = 1,000: the event hazards wrong (a Cox model on the treatment alone), with the default treatment learner and censoring library, which hold the true models; the targeted estimate | bias interval inside the equivalence margin | bias -0.000612 to 0.0023, margin 0.0045, SE ratio 1.0365 | pass |
| `double_robustness` | `l1_c1__plug_in_control` | control | the one-cause law, the treated arm's risk of the event by t = 1.48, ten outer folds, n = 1,000: the event hazards wrong (a Cox model on the treatment alone), with the default treatment learner and censoring library, which hold the true models; the initial plug-in of the same fit, before the walk | bias interval must fall entirely outside the margin | bias 0.0702 to 0.0729, margin 0.0042, SE ratio 1.1132 | pass |
| `double_robustness` | `l3_c1__factual_tilt_control` | control | the three-cause law, the treated arm's risk of cause 1 by t = 0.9, ten outer folds, n = 1,000: the event hazards wrong (a Cox model on the treatment alone), with the default treatment learner and censoring library, which hold the true models; the same draw fitted with tilt='factual', which tilts only each unit's received arm | bias interval must fall entirely outside the margin | bias 0.0269 to 0.0291, margin 0.0033, SE ratio 1.2202 | pass |
| `double_robustness` | `l3_c1__mechanism_correct` | positive | the three-cause law, the treated arm's risk of cause 1 by t = 0.9, ten outer folds, n = 1,000: the event hazards wrong (a Cox model on the treatment alone), with the default treatment learner and censoring library, which hold the true models; the targeted estimate | bias interval inside the equivalence margin | bias -0.000797 to 0.0014, margin 0.0035, SE ratio 1.0876 | pass |
| `double_robustness` | `l3_c1__plug_in_control` | control | the three-cause law, the treated arm's risk of cause 1 by t = 0.9, ten outer folds, n = 1,000: the event hazards wrong (a Cox model on the treatment alone), with the default treatment learner and censoring library, which hold the true models; the initial plug-in of the same fit, before the walk | bias interval must fall entirely outside the margin | bias 0.0411 to 0.0433, margin 0.0035, SE ratio 1.0949 | pass |
| `double_robustness` | `l3_c3__factual_tilt_control` | control | the three-cause law, the treated arm's risk of cause 3 by t = 0.9, ten outer folds, n = 1,000: the event hazards wrong (a Cox model on the treatment alone), with the default treatment learner and censoring library, which hold the true models; the same draw fitted with tilt='factual', which tilts only each unit's received arm | bias interval must fall entirely outside the margin | bias 0.0156 to 0.0176, margin 0.0030, SE ratio 1.3377 | pass |
| `double_robustness` | `l3_c3__mechanism_correct` | positive | the three-cause law, the treated arm's risk of cause 3 by t = 0.9, ten outer folds, n = 1,000: the event hazards wrong (a Cox model on the treatment alone), with the default treatment learner and censoring library, which hold the true models; the targeted estimate | bias interval inside the equivalence margin | bias -0.000854 to 0.0013, margin 0.0033, SE ratio 1.0770 | pass |
| `double_robustness` | `l3_c3__plug_in_control` | control | the three-cause law, the treated arm's risk of cause 3 by t = 0.9, ten outer folds, n = 1,000: the event hazards wrong (a Cox model on the treatment alone), with the default treatment learner and censoring library, which hold the true models; the initial plug-in of the same fit, before the walk | bias interval must fall entirely outside the margin | bias 0.0239 to 0.0259, margin 0.0030, SE ratio 1.1624 | pass |
| `interval_calibration` | `l1_rd__correctly_specified` | positive | the one-cause law, the difference in the risk of the event by t = 1.48 between treatment and no treatment, ten outer folds, n = 1,000: the default libraries, each of which holds the true model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9324 to 0.9537, SE ratio 0.9651 to 1.0294 | pass |
| `interval_calibration` | `l1_rd__noise_control` | control | the one-cause law, the difference in the risk of the event by t = 1.48 between treatment and no treatment, ten outer folds, n = 1,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8077 to 0.8426, SE ratio 0.6841 to 0.7281 | pass |
| `interval_calibration` | `l1_rd__shrunken_se_control` | control | the one-cause law, the difference in the risk of the event by t = 1.48 between treatment and no treatment, ten outer folds, n = 1,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8829 to 0.9109, SE ratio 0.8194 to 0.8746 | pass |
| `interval_calibration` | `l3_rd__correctly_specified` | positive | the three-cause law, the difference in the risk of cause 3 by t = 0.9 between treatment and no treatment, ten outer folds, n = 1,000: the default libraries, each of which holds the true model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9409 to 0.9608, SE ratio 0.9865 to 1.0557 | pass |
| `interval_calibration` | `l3_rd__noise_control` | control | the three-cause law, the difference in the risk of cause 3 by t = 0.9 between treatment and no treatment, ten outer folds, n = 1,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8204 to 0.8542, SE ratio 0.6950 to 0.7447 | pass |
| `interval_calibration` | `l3_rd__shrunken_se_control` | control | the three-cause law, the difference in the risk of cause 3 by t = 0.9 between treatment and no treatment, ten outer folds, n = 1,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.9002 to 0.9261, SE ratio 0.8374 to 0.8978 | pass |
<!-- /generated -->

The calibration cells read L1's risk difference and L3's cause-3 risk difference, the cause this
study adds, each at its law's median, on 3,200 draws: coverage 0.944 and SE ratio 0.996 for L1,
coverage 0.952 and SE ratio 1.019 for L3, inside both bands. The two controls are derived from
each cell's rows and leave the band as declared: the noise control, which adds one efficient
standard deviation of independent noise, reads SE ratio 0.70 and 0.72, and the shrunken-SE
control, which multiplies each standard error by 0.85, reads 0.85 and 0.87. The cells read the
coverage and SE-ratio bands only.

The double-robustness cells fit 1,600 draws per law with the wrong event hazards, the Cox model on
the treatment alone, and the right treatment and censoring models, which the default learners
hold. The targeted estimate is unbiased: standardized bias 0.037 for L1's treated risk at
$t = 1.48$, and 0.019 and 0.013 for L3's treated risks of causes 1 and 3 at $t = 0.9$. Two
controls on the same draws must stay biased. The initial plug-in, before the walk, reads 3.43,
2.44 and 1.63. The fit with `tilt="factual"`, which tilts only each unit's received arm as R
`concrete` 1.0.5 does, reads 2.30, 1.69 and 1.10. The plug-in has no standard error of its own,
so its row carries the targeted fit's and its verdict reads the bias only.
`tests/studies/continuous_survival_causes_properties.py` declares every cell.

## Reported readings

These readings carry no verdict.

### The first quartile

`early-time.csv` holds the first quartile's targets from the primary fits: three at $t = 0.58$
for L1 and nine at $t = 0.37$ for L3. Every estimate is unbiased (standardized bias within
0.044). L1's three cover 0.943 to 0.954 with SE ratios 1.015 to 1.023. L3's nine cover 0.924 to
0.962 with SE ratios 0.968 to 1.049. The lowest is L3's treated risk of cause 2, the sparsest
cell, at coverage 0.924 and SE ratio 0.968, the same cell that covered least among the gated
rows. A fit's support label reads experimental at a time this early, as it does with two causes.

### The walk

`targeting-readings.csv` gives, per law, the steps, the share of fits that met the stopping rule,
and the floor shares. Every L1 fit met the rule, in 2.6 steps on average and 7 at most. L3's fits
met it in all but one of 7,200, in 3.8 steps on average and 49 at most. The score equation held
at 33% (L1) and 46% (L3) of 100 random times on average, against 40% in the registered study.

## Sizing

Every size was fixed before the run from the pilot in
`tests/studies/continuous_survival_causes_pilot.py`, on seed streams no declared cell reads. The
primary count, 7,200, follows the shared rule of `tests/studies/repeated_sizing.py` with one
implementation, set by L3's control risk of cause 1 at its median, which covered 0.926 on 800
pilot draws. The calibration count, 3,200, follows the arm-pooled predictive rule on 2,000 pilot
draws per cell. The double-robustness count, 1,600 draws per law, passes the three positive cells
together with probability 0.997 at their pilot biases.
`tests/unit/test_continuous_survival_causes_design.py` recomputes each count.

## Red-cell route

The study is gated, and no cell is red. A red row would have kept its number of causes out of
`MEASURED_CAUSES`, which the design test ties to this record. The declared readings are in
`tests/canonical/continuous_survival_causes/regenerate.py`.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 7200 | primary replications |
| `n` | 1000 | observations per replication of the record's reference size |
| `independent_tests_passed` | 24 | truth tests passing |
| `independent_tests_total` | 24 | truth tests reported |
| `property_cells_passed` | 15 | property cells passing |
| `property_cells_total` | 15 | property cells reported |
| `max_standardized_bias` | 0.0343 | largest primary standardized bias |
| `min_coverage` | 0.9301 | lowest primary coverage |
| `properties[double_robustness/l1_c1__mechanism_correct]:standardized_bias` | 0.0372 | standardized bias of L1's targeted estimate under wrong hazards |
| `properties[double_robustness/l3_c3__mechanism_correct]:standardized_bias` | 0.0125 | standardized bias of L3's cause-3 targeted estimate under wrong hazards |
| `properties[double_robustness/l3_c3__plug_in_control]:standardized_bias` | 1.6336 | standardized bias of L3's cause-3 plug-in |
| `properties[double_robustness/l3_c3__factual_tilt_control]:standardized_bias` | 1.1042 | standardized bias of L3's cause-3 factual-tilt fit |
| `properties[interval_calibration/l1_rd__correctly_specified]:se_ratio` | 0.9959 | SE ratio of L1's calibration cell |
| `properties[interval_calibration/l1_rd__correctly_specified]:coverage` | 0.9437 | coverage of L1's calibration cell |
| `properties[interval_calibration/l3_rd__correctly_specified]:se_ratio` | 1.0194 | SE ratio of L3's calibration cell |
| `properties[interval_calibration/l3_rd__correctly_specified]:coverage` | 0.9516 | coverage of L3's calibration cell |
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
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.8500 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 72 | the same bound, in primary replications |

## Limits

| limit | what it means for use |
| --- | --- |
| one cause and three causes, each on one law | the evidence covers the base law with its second cause removed and with a third added. Four or more causes are measured by no study and read experimental |
| the default libraries, n = 1,000 | the evidence covers the default Cox libraries and treatment learner and the default settings, cross-fitted. A fit with one or three causes reads supported from 1,000 units (`CAUSES_N`): the 500-unit floor of the [switched-variance study](continuous-survival-variance.md) was measured with two causes |
| the median and upper quartile | the gated times sit at about 50% and 75% of the observed events, and the time condition of the [continuous-time survival study](continuous-time-survival.md#limits) applies. The first quartile is published without a verdict |
| risks and risk differences, no restricted horizon | the [restricted-means study](continuous-survival-functionals.md) measures restricted horizons with two causes only, so a fit with one or three causes and a restricted horizon reads experimental, as do risk ratios and the band |
| no comparator | no maintained implementation of this construction was located |

## Reproduction

Run the pilot first, as `python -m tests.studies.continuous_survival_causes_pilot --jobs N`.
Run `python -m tests.canonical.continuous_survival_causes.regenerate --output DIR` with an empty
scratch directory outside the repository to regenerate the artifacts. It runs no reference
process, because the study has no comparator. The red-cell readings and the budget are declared in
`tests/canonical/continuous_survival_causes/regenerate.py` before any run.
