# Score assessment by TMLE

This study validates `IdentifiedEffect.assess_score(...)`, which assesses a conditional-effect
score fitted on other data. For one score it reports the risk difference against the best
constant and the calibration intercept and slope; with `compare=` it also reports the paired
risk difference $R(s_1) - R(s_2)$ of two scores. Every estimate is the substitution estimate of
one pooled TMLE fluctuation of the cross-fitted ATE fit along the span of the scores' features,
with a Wald interval from its influence curve. Both the difference scale and the log-ratio scale
are measured. [External scores](../conditional-effects.md#external-scores) gives the estimands,
the estimator and its conditions.

**No canonical implementation is compared.** EconML 0.16.0 comes closest. Its `RScorer`
returns a point score per CATE model, an R-square analogue of the R-loss against a
constant-effect baseline, with no interval. Its `DRTester.evaluate_blp` regresses doubly robust
pseudo-outcomes on one model's predictions by OLS and reports the slope with a standard error
and a p-value (the classical OLS error for a binary treatment, HC1 for each arm of a categorical
one): one score's calibration slope on the difference scale, from a one-step regression rather
than a TMLE. Neither reports a risk difference against the best constant, a paired risk
difference of two scores or a log-ratio calibration, and no comparison was run. The absence
does not block the row, and a zero-row equivalence artifact records it.

## What was tested

| setting | declaration |
| --- | --- |
| laws | $W \sim U(-1, 1)^3$. Binary laws of the `conditional-projection` study, with $\operatorname{logit} \mu(a, W) = -0.3 + 0.6 W_1 - 0.4 W_2 + a(0.3 + 0.5 W_1 + 0.4 \sin \pi W_3)$: `observational` ($\operatorname{logit} g = 0.4 W_1 - 0.5 W_2 + 0.3 W_3$), `trial` ($g = 0.5$, declared known) and `noweight_law` ($\operatorname{logit} g = 2.5 W_1$). Law `H` of the `targeting-curves` study: a continuous outcome with effect $1 + W_1$ |
| scores | fixed functions of $W_1$ and $W_3$. Binary paired groups: $S = 0.2 + 0.3 W_1$ against $S_B = S + 0.15 \sin \pi W_3$ on the difference scale, and $S = 0.15 + 0.25 W_1$ against $S + 0.2 \sin \pi W_3$ on the log-ratio scale. Law `H`: $S = 1 + 0.8 W_1$ against $S + 0.3 W_3$; two scores of equal risk, $1 + 1.5 W_1$ and $1 + W_1 + 0.5 W_3$; and $1 + W_1 + 0.13 W_3$ against $1 + W_1$ for power. The `dr` groups compare the first binary score with a quadratic in $W_1$, the shape of the declared outcome error |
| groups | nine fit groups: `paired_2k` (`observational`), `paired_trial_2k` (`trial`), `paired_cont_2k`, `equal_cont_2k` and `power_cont_2k` (law `H`), each at n = 2,000 on the difference scale; `single_crr_4k` and `paired_crr_4k` (`observational`, log-ratio scale) at n = 4,000; and `paired_dr_4k` and `paired_crr_dr_4k` (`noweight_law`, one per scale) at n = 4,000 |
| fit | `effect.assess_score("S", computed_from=[...], score_kind="external", compare="SB", compare_computed_from=[...])` on `ConditionalContrast(Modifiers(["W1", "W3"]), scale=...)`, without `compare=` on `single_crr_4k`, with every other setting at its default |
| learners | a spline GLM per arm (cubic splines, five knots) for $\mu$ and a logistic GLM for $g$. The two `dr` groups fit a declared wrong outcome regression, the true logit plus $1.5(W_1^2 - 1/3)$ in arm 1 and $-W_1$ in arm 0, with the logistic GLM for $g$ |
| truth | Gauss-Legendre quadrature with 48 nodes per axis; exact on law `H`, where every integrand is a polynomial of degree at most 4 |
| primary budget | 2,000 replications of `paired_2k` at n = 2,000, on the paired risk difference |
| primary verdicts | the framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| property budget | 15,200 replications of each calibration group, 1,000 of `power_cont_2k` and 2,000 of each `dr` group |
| seeds | `seed=20261297` and `resampling_seed=20261298`; every group draws from its own stream, keyed by its label |
| policy | `gated` |

The laws, groups, learners, budgets, controls and seeds were fixed before the run, in
`tests/studies/score_assessment.py`. The calibration count of 15,200 is the pooled predictive
rule at the SE ratios of a 2,400-draw pilot on a disjoint stream
(`tests/studies/score_assessment_pilot.csv`). The log-ratio risk differences of a single score
form their own pool, sized against an oracle computation with the true nuisances
(`tests/studies/score_assessment_oracle.csv`, 40,000 draws). Each positive bias row is priced
at the larger of its second-order theory bias and its oracle bias, which sets 2,000.
`tests/unit/test_score_assessment_design.py` checks the truths, the cells and those design
numbers.

**Declaration history.** The study was first declared at commit `3c00484`. Before any run it was
redeclared under the design review's rev 2.2. Those changes added the reduced-span retry of the
log-ratio fluctuation, the pilot at 2,400 draws, the separate pool and the oracle reference.
The n rule moved both log-ratio calibration groups from 2,000 to 4,000 rows. The final
declaration is commit `804cfea`, which raised the bias budget to 2,000, and the declared run
ran at that commit. `tests/canonical/score_assessment/regenerate.py` lists every change.

**Why the log-ratio rows run at n = 4,000.** At 2,000 rows the pilot's pooled SE ratio on the
log-ratio risk differences was 1.047, within the 99% interval of the oracle's 1.026. That 1.047
is the oracle's best-constant term plus the unexplained excess below (R1). At that ratio the
sizing predicted that no replication count up to 50,000 meets the family-wise calibration rule
(0.920 at the cap). The pre-run n rule (`score_assessment.n_rule`) therefore moved both groups
to 4,000 rows, with a fresh pilot and oracle on new streams. No interval was measured at 2,000
rows; the move rests on that prediction.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the observational binary law, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, a spline GLM per arm and a logistic treatment model, n = 2,000 | `risk_difference[S - SB]` | the paired risk difference: S's mean squared error for the CATE minus SB's | `cleverly` score assessment: one pooled TMLE fluctuation of the cross-fitted ATE fit along the span of the scores' features, with substitution estimates of each score's risk difference, calibration and the paired risk difference, and influence-curve Wald intervals | -0.000197 to 0.000349 | 0.9490 | 0.9996 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | for each parameter of the six calibration groups (R = 15,200), does the SE-ratio interval lie inside 0.93 to 1.07 and the coverage interval inside 0.92 to 0.98 | the paired row with the SE $\sqrt{SE_1^2 + SE_2^2}$ of the two single risk differences, which ignores their covariance, must lie above the band on `paired_2k` and `paired_cont_2k`; the paired row with the SE of its plug-in part alone, without the residual term, must lie below it on `equal_cont_2k` |
| `bias` | under the declared wrong outcome regression with an estimated $g$ (R = 2,000), is each risk difference's standardized-bias interval inside $\pm 0.25$ | the plug-in at that initial outcome regression, without targeting, must lie wholly outside $\pm 0.25$ |
| `type_i_error` | on `equal_cont_2k`, where the two risks are equal, is the 99% upper end of the paired Wald test's rejection rate at most 0.10 | paired with the power cell |
| `power` | on `power_cont_2k` (R = 1,000), is the lower end of the paired test's rejection rate at least 0.80 | an inert test fails it |

**The best-constant term.** The risk difference of a single score is $R(s) - R(c_n)$ at the
estimated best constant $c_n$. Because $c_n$ minimizes the estimated risk among constants, the
row is biased upward by about $\operatorname{Var}(R_n'(c_0))/(2R''(c_0))$, which is $O(1/n)$,
and its influence curve evaluated at $c_n$ overstates the SE slightly. Both are second order and
vanish as $n$ grows. The paired row carries neither, because the constant cancels. The design
priced the term before the run: `score_assessment.theory_bias` predicts 0.118, 0.118 and 0.084
SD on the three log-ratio risk differences at n = 4,000 and 0.051 SD on the difference scale at
2,000. The run measured 0.111, 0.127 and 0.088 SD on the log-ratio rows and 0.051 on
`paired_2k`'s $S$ row, with every calibration interval inside the band.

**The pilot's excess over the oracle (design re-review R1).** On the three log-ratio risk
differences the pilot's SE ratio exceeded the oracle's by about 0.024 at 2,000 rows and 0.021
at 4,000. The oracle's own excess over 1 halves as $n$ doubles, as the best-constant term
predicts, and this gap does not, so the term does not explain it. No defect was found before
the run. The sizing read the pilot's ratios, so the calibration count of 15,200 prices the
excess. In the run the three rows read SE ratios of 1.004, 1.017 and 1.006 against the oracle's
1.015, 1.013 and 1.004, within about 0.01 of it, so the run does not reproduce the pilot's 0.02
excess. Its source is not established.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `bias` | `paired_crr_dr_4k_SB_risk_difference` | positive | the wide-overlap binary law with a declared wrong outcome regression, log-ratio scale, S = 0.15 + 0.25 W1 against SB = 0.05 + 0.1 W1 + 0.4 W1^2, n = 4,000: the TMLE under the declared wrong outcome regression and a fitted mechanism, SB's risk minus the risk of the best constant | the 99% interval of the standardized bias inside +-0.25 | bias 0.000042 to 0.000254, margin 0.000460 | pass |
| `bias` | `paired_crr_dr_4k_S_risk_difference` | positive | the wide-overlap binary law with a declared wrong outcome regression, log-ratio scale, S = 0.15 + 0.25 W1 against SB = 0.05 + 0.1 W1 + 0.4 W1^2, n = 4,000: the TMLE under the declared wrong outcome regression and a fitted mechanism, S's risk minus the risk of the best constant | the 99% interval of the standardized bias inside +-0.25 | bias 0.000079 to 0.000282, margin 0.000439 | pass |
| `bias` | `paired_crr_dr_4k_paired` | positive | the wide-overlap binary law with a declared wrong outcome regression, log-ratio scale, S = 0.15 + 0.25 W1 against SB = 0.05 + 0.1 W1 + 0.4 W1^2, n = 4,000: the TMLE under the declared wrong outcome regression and a fitted mechanism, S's risk minus SB's risk | the 99% interval of the standardized bias inside +-0.25 | bias -0.000066 to 0.000130, margin 0.000425 | pass |
| `bias` | `paired_dr_4k_SB_risk_difference` | positive | the wide-overlap binary law with a declared wrong outcome regression, difference scale, S = 0.2 + 0.3 W1 against SB = 0.1 + 0.1 W1 + 0.4 W1^2, n = 4,000: the TMLE under the declared wrong outcome regression and a fitted mechanism, SB's risk minus the risk of the best constant | the 99% interval of the standardized bias inside +-0.25 | bias -0.000665 to 0.000583, margin 0.0027 | pass |
| `bias` | `paired_dr_4k_S_risk_difference` | positive | the wide-overlap binary law with a declared wrong outcome regression, difference scale, S = 0.2 + 0.3 W1 against SB = 0.1 + 0.1 W1 + 0.4 W1^2, n = 4,000: the TMLE under the declared wrong outcome regression and a fitted mechanism, S's risk minus the risk of the best constant | the 99% interval of the standardized bias inside +-0.25 | bias -0.000235 to 0.000908, margin 0.0025 | pass |
| `bias` | `paired_dr_4k_paired` | positive | the wide-overlap binary law with a declared wrong outcome regression, difference scale, S = 0.2 + 0.3 W1 against SB = 0.1 + 0.1 W1 + 0.4 W1^2, n = 4,000: the TMLE under the declared wrong outcome regression and a fitted mechanism, S's risk minus SB's risk | the 99% interval of the standardized bias inside +-0.25 | bias -0.000090 to 0.000845, margin 0.0020 | pass |
| `bias` | `untargeted_crr_dr_4k_SB_risk_difference` | control | the wide-overlap binary law with a declared wrong outcome regression, log-ratio scale, S = 0.15 + 0.25 W1 against SB = 0.05 + 0.1 W1 + 0.4 W1^2, n = 4,000: the plug-in at the declared wrong outcome regression, with no targeting, SB's risk minus the risk of the best constant | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0093 to -0.0092, margin 0.000071 | pass |
| `bias` | `untargeted_crr_dr_4k_S_risk_difference` | control | the wide-overlap binary law with a declared wrong outcome regression, log-ratio scale, S = 0.15 + 0.25 W1 against SB = 0.05 + 0.1 W1 + 0.4 W1^2, n = 4,000: the plug-in at the declared wrong outcome regression, with no targeting, S's risk minus the risk of the best constant | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0100 to -0.0099, margin 0.000050 | pass |
| `bias` | `untargeted_crr_dr_4k_paired` | control | the wide-overlap binary law with a declared wrong outcome regression, log-ratio scale, S = 0.15 + 0.25 W1 against SB = 0.05 + 0.1 W1 + 0.4 W1^2, n = 4,000: the plug-in at the declared wrong outcome regression, with no targeting, S's risk minus SB's risk | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.000705 to -0.000685, margin 0.000043 | pass |
| `bias` | `untargeted_dr_4k_SB_risk_difference` | control | the wide-overlap binary law with a declared wrong outcome regression, difference scale, S = 0.2 + 0.3 W1 against SB = 0.1 + 0.1 W1 + 0.4 W1^2, n = 4,000: the plug-in at the declared wrong outcome regression, with no targeting, SB's risk minus the risk of the best constant | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0353 to -0.0351, margin 0.000360 | pass |
| `bias` | `untargeted_dr_4k_S_risk_difference` | control | the wide-overlap binary law with a declared wrong outcome regression, difference scale, S = 0.2 + 0.3 W1 against SB = 0.1 + 0.1 W1 + 0.4 W1^2, n = 4,000: the plug-in at the declared wrong outcome regression, with no targeting, S's risk minus the risk of the best constant | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0433 to -0.0431, margin 0.000289 | pass |
| `bias` | `untargeted_dr_4k_paired` | control | the wide-overlap binary law with a declared wrong outcome regression, difference scale, S = 0.2 + 0.3 W1 against SB = 0.1 + 0.1 W1 + 0.4 W1^2, n = 4,000: the plug-in at the declared wrong outcome regression, with no targeting, S's risk minus SB's risk | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0080 to -0.0079, margin 0.000275 | pass |
| `interval_calibration` | `equal_cont_2k_paired__correctly_specified` | positive | law H, S = 1 + 1.5 W1 against SB = 1 + W1 + 0.5 W3, whose risks are equal, n = 2,000: S's risk minus SB's risk: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9451 to 0.9543, SE ratio 0.9801 to 1.0089 | pass |
| `interval_calibration` | `equal_cont_2k_paired__term_dropped_control` | control | law H, S = 1 + 1.5 W1 against SB = 1 + W1 + 0.5 W3, whose risks are equal, n = 2,000: S's risk minus SB's risk: the same paired estimate with the SE of its influence curve's plug-in part alone, without the residual term | the SE-ratio interval must lie below the calibration band | coverage 0.3834 to 0.4039, SE ratio 0.2590 to 0.2666 | pass |
| `interval_calibration` | `paired_2k_SB_calibration_intercept__correctly_specified` | positive | the observational binary law, difference scale, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), n = 2,000: the intercept of SB's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9442 to 0.9535, SE ratio 0.9858 to 1.0161 | pass |
| `interval_calibration` | `paired_2k_SB_calibration_slope__correctly_specified` | positive | the observational binary law, difference scale, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), n = 2,000: the slope of SB's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9468 to 0.9559, SE ratio 0.9854 to 1.0153 | pass |
| `interval_calibration` | `paired_2k_SB_risk_difference__correctly_specified` | positive | the observational binary law, difference scale, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), n = 2,000: SB's risk minus the risk of the best constant: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9455 to 0.9547, SE ratio 0.9804 to 1.0096 | pass |
| `interval_calibration` | `paired_2k_S_calibration_intercept__correctly_specified` | positive | the observational binary law, difference scale, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), n = 2,000: the intercept of S's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9454 to 0.9546, SE ratio 0.9855 to 1.0158 | pass |
| `interval_calibration` | `paired_2k_S_calibration_slope__correctly_specified` | positive | the observational binary law, difference scale, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), n = 2,000: the slope of S's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9451 to 0.9543, SE ratio 0.9849 to 1.0146 | pass |
| `interval_calibration` | `paired_2k_S_risk_difference__correctly_specified` | positive | the observational binary law, difference scale, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), n = 2,000: S's risk minus the risk of the best constant: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9429 to 0.9522, SE ratio 0.9801 to 1.0098 | pass |
| `interval_calibration` | `paired_2k_paired__correctly_specified` | positive | the observational binary law, difference scale, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), n = 2,000: S's risk minus SB's risk: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9434 to 0.9527, SE ratio 0.9815 to 1.0114 | pass |
| `interval_calibration` | `paired_2k_paired__inflated_se_control` | control | the observational binary law, difference scale, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), n = 2,000: S's risk minus SB's risk: the same paired estimate with the SE sqrt(SE_S^2 + SE_SB^2) of the two single-score risk differences, which ignores their covariance | the SE-ratio interval must lie above the calibration band | coverage 0.9997 to 1, SE ratio 2.9807 to 3.0711 | pass |
| `interval_calibration` | `paired_cont_2k_SB_calibration_intercept__correctly_specified` | positive | law H (continuous outcome, effect 1 + W1), S = 1 + 0.8 W1 against SB = S + 0.3 W3, n = 2,000: the intercept of SB's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9442 to 0.9534, SE ratio 0.9776 to 1.0068 | pass |
| `interval_calibration` | `paired_cont_2k_SB_calibration_slope__correctly_specified` | positive | law H (continuous outcome, effect 1 + W1), S = 1 + 0.8 W1 against SB = S + 0.3 W3, n = 2,000: the slope of SB's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9447 to 0.9539, SE ratio 0.9774 to 1.0068 | pass |
| `interval_calibration` | `paired_cont_2k_SB_risk_difference__correctly_specified` | positive | law H (continuous outcome, effect 1 + W1), S = 1 + 0.8 W1 against SB = S + 0.3 W3, n = 2,000: SB's risk minus the risk of the best constant: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9429 to 0.9523, SE ratio 0.9785 to 1.0077 | pass |
| `interval_calibration` | `paired_cont_2k_S_calibration_intercept__correctly_specified` | positive | law H (continuous outcome, effect 1 + W1), S = 1 + 0.8 W1 against SB = S + 0.3 W3, n = 2,000: the intercept of S's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9432 to 0.9526, SE ratio 0.9779 to 1.0068 | pass |
| `interval_calibration` | `paired_cont_2k_S_calibration_slope__correctly_specified` | positive | law H (continuous outcome, effect 1 + W1), S = 1 + 0.8 W1 against SB = S + 0.3 W3, n = 2,000: the slope of S's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9436 to 0.9529, SE ratio 0.9771 to 1.0069 | pass |
| `interval_calibration` | `paired_cont_2k_S_risk_difference__correctly_specified` | positive | law H (continuous outcome, effect 1 + W1), S = 1 + 0.8 W1 against SB = S + 0.3 W3, n = 2,000: S's risk minus the risk of the best constant: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9428 to 0.9522, SE ratio 0.9781 to 1.0080 | pass |
| `interval_calibration` | `paired_cont_2k_paired__correctly_specified` | positive | law H (continuous outcome, effect 1 + W1), S = 1 + 0.8 W1 against SB = S + 0.3 W3, n = 2,000: S's risk minus SB's risk: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9437 to 0.9530, SE ratio 0.9827 to 1.0125 | pass |
| `interval_calibration` | `paired_cont_2k_paired__inflated_se_control` | control | law H (continuous outcome, effect 1 + W1), S = 1 + 0.8 W1 against SB = S + 0.3 W3, n = 2,000: S's risk minus SB's risk: the same paired estimate with the SE sqrt(SE_S^2 + SE_SB^2) of the two single-score risk differences, which ignores their covariance | the SE-ratio interval must lie above the calibration band | coverage 0.9997 to 1, SE ratio 4.3156 to 4.4472 | pass |
| `interval_calibration` | `paired_crr_4k_SB_calibration_intercept__correctly_specified` | positive | the observational binary law, log-ratio scale, S = 0.15 + 0.25 W1 against SB = S + 0.2 sin(pi W3), n = 4,000: the intercept of SB's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9429 to 0.9522, SE ratio 0.9793 to 1.0083 | pass |
| `interval_calibration` | `paired_crr_4k_SB_calibration_slope__correctly_specified` | positive | the observational binary law, log-ratio scale, S = 0.15 + 0.25 W1 against SB = S + 0.2 sin(pi W3), n = 4,000: the slope of SB's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9444 to 0.9536, SE ratio 0.9823 to 1.0121 | pass |
| `interval_calibration` | `paired_crr_4k_SB_risk_difference__correctly_specified` | positive | the observational binary law, log-ratio scale, S = 0.15 + 0.25 W1 against SB = S + 0.2 sin(pi W3), n = 4,000: SB's risk minus the risk of the best constant: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9479 to 0.9569, SE ratio 0.9914 to 1.0213 | pass |
| `interval_calibration` | `paired_crr_4k_S_calibration_intercept__correctly_specified` | positive | the observational binary law, log-ratio scale, S = 0.15 + 0.25 W1 against SB = S + 0.2 sin(pi W3), n = 4,000: the intercept of S's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9458 to 0.9549, SE ratio 0.9828 to 1.0126 | pass |
| `interval_calibration` | `paired_crr_4k_S_calibration_slope__correctly_specified` | positive | the observational binary law, log-ratio scale, S = 0.15 + 0.25 W1 against SB = S + 0.2 sin(pi W3), n = 4,000: the slope of S's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9468 to 0.9559, SE ratio 0.9875 to 1.0171 | pass |
| `interval_calibration` | `paired_crr_4k_S_risk_difference__correctly_specified` | positive | the observational binary law, log-ratio scale, S = 0.15 + 0.25 W1 against SB = S + 0.2 sin(pi W3), n = 4,000: S's risk minus the risk of the best constant: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9488 to 0.9577, SE ratio 1.0021 to 1.0324 | pass |
| `interval_calibration` | `paired_crr_4k_paired__correctly_specified` | positive | the observational binary law, log-ratio scale, S = 0.15 + 0.25 W1 against SB = S + 0.2 sin(pi W3), n = 4,000: S's risk minus SB's risk: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9434 to 0.9527, SE ratio 0.9818 to 1.0110 | pass |
| `interval_calibration` | `paired_trial_2k_SB_calibration_intercept__correctly_specified` | positive | the randomized binary law with P(A = 1) = 0.5 declared known, difference scale, the scores of paired_2k, n = 2,000: the intercept of SB's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9451 to 0.9543, SE ratio 0.9901 to 1.0197 | pass |
| `interval_calibration` | `paired_trial_2k_SB_calibration_slope__correctly_specified` | positive | the randomized binary law with P(A = 1) = 0.5 declared known, difference scale, the scores of paired_2k, n = 2,000: the slope of SB's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9453 to 0.9544, SE ratio 0.9858 to 1.0143 | pass |
| `interval_calibration` | `paired_trial_2k_SB_risk_difference__correctly_specified` | positive | the randomized binary law with P(A = 1) = 0.5 declared known, difference scale, the scores of paired_2k, n = 2,000: SB's risk minus the risk of the best constant: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9445 to 0.9538, SE ratio 0.9866 to 1.0172 | pass |
| `interval_calibration` | `paired_trial_2k_S_calibration_intercept__correctly_specified` | positive | the randomized binary law with P(A = 1) = 0.5 declared known, difference scale, the scores of paired_2k, n = 2,000: the intercept of S's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9452 to 0.9544, SE ratio 0.9901 to 1.0195 | pass |
| `interval_calibration` | `paired_trial_2k_S_calibration_slope__correctly_specified` | positive | the randomized binary law with P(A = 1) = 0.5 declared known, difference scale, the scores of paired_2k, n = 2,000: the slope of S's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9438 to 0.9531, SE ratio 0.9843 to 1.0138 | pass |
| `interval_calibration` | `paired_trial_2k_S_risk_difference__correctly_specified` | positive | the randomized binary law with P(A = 1) = 0.5 declared known, difference scale, the scores of paired_2k, n = 2,000: S's risk minus the risk of the best constant: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9447 to 0.9539, SE ratio 0.9845 to 1.0149 | pass |
| `interval_calibration` | `paired_trial_2k_paired__correctly_specified` | positive | the randomized binary law with P(A = 1) = 0.5 declared known, difference scale, the scores of paired_2k, n = 2,000: S's risk minus SB's risk: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9435 to 0.9528, SE ratio 0.9917 to 1.0210 | pass |
| `interval_calibration` | `single_crr_4k_S_calibration_intercept__correctly_specified` | positive | the observational binary law, log-ratio scale, S = 0.15 + 0.25 W1 alone, n = 4,000: the intercept of S's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9464 to 0.9555, SE ratio 0.9808 to 1.0100 | pass |
| `interval_calibration` | `single_crr_4k_S_calibration_slope__correctly_specified` | positive | the observational binary law, log-ratio scale, S = 0.15 + 0.25 W1 alone, n = 4,000: the slope of S's calibration line: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9431 to 0.9525, SE ratio 0.9747 to 1.0037 | pass |
| `interval_calibration` | `single_crr_4k_S_risk_difference__correctly_specified` | positive | the observational binary law, log-ratio scale, S = 0.15 + 0.25 W1 alone, n = 4,000: S's risk minus the risk of the best constant: the influence-curve Wald interval, with a spline GLM per arm (a linear spline on law H) and a logistic treatment model, or the known mechanism on the randomized law | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9456 to 0.9548, SE ratio 0.9891 to 1.0187 | pass |
| `power` | `paired_wald__c_0.13` | positive | law H: the two-sided 95% Wald test that S's risk minus SB's is 0, n = 2,000: S = 1 + W1 + 0.13 W3 against SB = 1 + W1, where the test's power is 0.90 | the rejection rate's lower end at least 0.80 | rejection 0.8960, 0.8687 to 0.9194 | pass |
| `type_i_error` | `paired_wald__equal_risks` | positive | law H: the two-sided 95% Wald test that S's risk minus SB's is 0, n = 2,000: S = 1 + 1.5 W1 against SB = 1 + W1 + 0.5 W3, whose risks are equal | the 99% upper end of the rejection rate at most 0.10 | rejection 0.0502, 0.0457 to 0.0549 | pass |
<!-- /generated -->

## Result

Every primary test passed, and so did every property cell. No fit failed. The 32 calibration
positives have SE-ratio intervals inside 0.93 to 1.07 and coverage intervals inside 0.92 to
0.98, on both scales, on binary and continuous outcomes, with an estimated and a known
mechanism, and on the paired row of two scores of equal risk. Each control fails as declared.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 1 of 1 | `performance-tests.csv` |
| property cells passed | 49 of 49 | `properties.csv` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |

The calibration positives read SE ratios from 0.989 to 1.017 and coverage from 0.948 to 0.953.
The paired Wald test rejects in 0.050 of the replications where the risks are equal and in
0.896 at the declared difference of `power_cont_2k`. Under the declared wrong outcome
regression every TMLE risk difference stays within 0.11 SD of the truth, while the untargeted
plug-in lies 4.0 to 49 SD away. The SE that ignores the covariance of the two scores reads SE
ratios of 3.03 and 4.38, and the SE without the residual term 0.263.

**Reduced-span fits.** A log-ratio fit whose alternation does not settle reruns along the span
without its thinnest direction, and is accepted only when every original score equation still
holds (see [External scores](../conditional-effects.md#external-scores)). `comparison.csv`
counts these fits by group and their coverage, ungated:

| group | reduced-span fits | coverage on those fits |
| --- | --- | --- |
| `single_crr_4k` | 27 of 15,200 | 1.000, 0.963 and 1.000 on the risk difference, intercept and slope |
| `paired_crr_4k` | 49 of 15,200 | 0.918 on each $S$ row; 0.898, 0.939 and 0.959 on the $S_B$ rows; 0.857 on the paired row |
| `paired_crr_dr_4k` | 155 of 2,000 | 0.981, 0.942 and 0.974 on the $S$ rows; 0.981, 0.968 and 0.974 on the $S_B$ rows; 0.948 on the paired row |

No difference-scale fit reduces its span. These fits are a selected set, the draws on which the
full span would not settle, so nominal coverage is not expected on them, and such a fit reads
`experimental`. The paired row of `paired_crr_4k` covers on 42 of its 49 (99% Clopper-Pearson
interval 0.685 to 0.957, one-sided binomial p of 0.011 against 0.95), which is consistent with
under-coverage on that subset. The gated cells include these fits: 0.18% and 0.32% of the
supported log-ratio groups' replications, and 7.8% of `paired_crr_dr_4k`'s gated bias cells.
Removing 49 fits from 15,200 moves a coverage by at most 0.003, so no calibration verdict rests
on them.

The status rule was declared before the run. Every gated verdict passed, so an assessment whose
configuration a measured group covers reads `supported`. The measured configurations are
`cleverly.effect_curves.SCORE_ASSESSMENT_MEASURED` for the single-score rows and
`SCORE_PAIRED_MEASURED` for the paired row; every other configuration reads `experimental`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2000 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 1 | parameter tests against the truth |
| `independent_tests_passed` | 1 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 49 | repeated-sampling property cells |
| `property_cells_passed` | 49 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0161 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9490 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9350 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9558 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0463 | highest bootstrap primary SE-ratio endpoint |
| `properties[interval_calibration/single_crr_4k_S_risk_difference__correctly_specified]:se_ratio` | 1.0040 | single-score log-ratio risk difference, SE ratio |
| `properties[interval_calibration/single_crr_4k_S_risk_difference__correctly_specified]:standardized_bias` | 0.1113 | its standardized bias, the best-constant term |
| `properties[interval_calibration/paired_crr_4k_S_risk_difference__correctly_specified]:se_ratio` | 1.0170 | paired group's log-ratio $S$ risk difference, SE ratio |
| `properties[interval_calibration/paired_crr_4k_S_risk_difference__correctly_specified]:se_ratio_ci_upper` | 1.0324 | highest calibration SE-ratio endpoint |
| `properties[interval_calibration/paired_crr_4k_S_risk_difference__correctly_specified]:standardized_bias` | 0.1266 | its standardized bias |
| `properties[interval_calibration/paired_crr_4k_SB_risk_difference__correctly_specified]:se_ratio` | 1.0061 | log-ratio $S_B$ risk difference, SE ratio |
| `properties[interval_calibration/paired_crr_4k_SB_risk_difference__correctly_specified]:standardized_bias` | 0.0883 | its standardized bias |
| `properties[interval_calibration/single_crr_4k_S_calibration_slope__correctly_specified]:se_ratio_ci_lower` | 0.9747 | lowest calibration SE-ratio endpoint |
| `properties[interval_calibration/paired_crr_4k_paired__correctly_specified]:coverage` | 0.9482 | log-ratio paired row, coverage |
| `properties[interval_calibration/paired_2k_paired__correctly_specified]:coverage` | 0.9482 | difference-scale paired row, coverage |
| `properties[interval_calibration/paired_2k_S_risk_difference__correctly_specified]:standardized_bias` | 0.0506 | difference-scale $S$ risk difference, standardized bias |
| `properties[interval_calibration/equal_cont_2k_paired__correctly_specified]:coverage` | 0.9498 | paired row at equal risks, coverage |
| `properties[type_i_error/paired_wald__equal_risks]:rejection_rate` | 0.0502 | paired test, size |
| `properties[power/paired_wald__c_0.13]:rejection_rate` | 0.8960 | paired test, power |
| `properties[interval_calibration/paired_2k_paired__inflated_se_control]:se_ratio` | 3.0251 | SE ignoring the covariance, binary law |
| `properties[interval_calibration/paired_cont_2k_paired__inflated_se_control]:se_ratio` | 4.3791 | SE ignoring the covariance, law `H` |
| `properties[interval_calibration/equal_cont_2k_paired__term_dropped_control]:se_ratio` | 0.2628 | SE without the residual term |
| `properties[bias/paired_crr_dr_4k_S_risk_difference]:standardized_bias` | 0.1027 | wrong outcome regression, log-ratio $S$ risk difference |
| `properties[bias/paired_dr_4k_paired]:standardized_bias` | 0.0465 | wrong outcome regression, difference-scale paired row |
| `properties[bias/untargeted_crr_dr_4k_paired]:standardized_bias` | -4.0293 | untargeted plug-in, smallest displacement |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the primary coverage lower endpoint must clear |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
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
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:subject_failures_allowed` | 20 | failed primary fits that bound allows |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | EconML 0.16.0's `DRTester` reports one score's difference-scale calibration slope from a one-step OLS regression, but no published implementation reports a score's risk difference or a paired risk, and no comparison was run. The row rests on accuracy against the truth and on the theory properties |
| The measured configurations | A binary outcome with an estimated mechanism on either scale, a binary outcome with a known mechanism on the difference scale, and a continuous outcome with an estimated mechanism on the difference scale, each with a binary treatment, iid unweighted rows, complete outcomes, one split and cross-fitted nuisances. The paired row is measured in the same four. Weights, clusters, missing outcomes, a pair of arms of a multi-arm treatment, an ATE fit without cross-fitting, and every other scale, outcome and mechanism read `experimental` |
| Sizes, laws and learners | The difference-scale groups run at n = 2,000 and the log-ratio groups at 4,000; at 2,000 rows the pilot's SE ratio of 1.047 (the best-constant term plus the R1 excess) predicted that no count up to 50,000 meets the calibration rule, and no interval was measured there. Every law is smooth on $[-1, 1]^3$, and the nuisances are spline or logistic GLMs; the default learner library is not measured. The status does not read $n$ or the learner, so such a fit within a measured configuration reads `supported` and rests on the theory |
| The best-constant term | A single score's risk difference is biased upward by a second-order term, measured with consistent nuisances at 0.09 to 0.13 SD on the calibration groups' log-ratio rows at n = 4,000 (0.08 to 0.10 on the `dr` rows) and 0.03 to 0.05 SD on the difference scale at 2,000. It is largest when the score is close to the best constant. The paired row does not carry it |
| The pilot's unexplained excess (R1) | On the log-ratio risk differences the pilot's SE ratio exceeded the oracle's by about 0.02 at both sizes, which the best-constant term does not explain. The run did not reproduce it, and its source is not established |
| Reduced-span fits | 231 log-ratio fits settled only on a reduced span. Their coverage ranged from 0.857 to 1.000 by row, and the paired row's 42 of 49 is consistent with under-coverage on these selected draws. Such a fit reads `experimental`; the gated cells include these fits, and no verdict rests on them. Coverage at a true calibration slope near 0 or 0.5, where the retry keeps firing as $n$ grows, is not measured: the study's true slopes are 0.73, 0.83 and 0.26 |
| One check is blind on law `H` | On law `H` the effect $1 + W_1$ is linear in $W_1$, so for a score linear in $W_1$ (`paired_cont_2k`'s and `equal_cont_2k`'s $S$) the plug-in part of the calibration's influence curve is identically 0, and an error in it would go unseen there. The binary groups, whose effect is not in the span of $(1, S)$, measure that part |
| One wrong nuisance | The `dr` groups gate the bias under a wrong outcome regression with an estimated $g$, and `comparison.csv` publishes their coverage ungated. An interval under one inconsistent nuisance is not claimed |
| The scores and their selection | Every score is a fixed function fitted outside the sample. A score fitted on these rows, out-of-fold predictions included, is not external, and the interval covers one pair named before the fit, not the best of several |

## Reproduction

The declared run is `python -m tests.canonical.score_assessment.regenerate`, whose docstring
gives the run form, the changes since the first declaration and the red-cell reading. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/score_assessment/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/score_assessment/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/score_assessment/replicates.parquet),
the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/score_assessment/properties.csv)
and the [comparison with the one-step](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/score_assessment/comparison.csv)
carry every published row.
