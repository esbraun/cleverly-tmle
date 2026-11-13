# Working model over the hazard curve

This study validates longitudinal TMLE with `msm=`, `survival_model="hazard"` and
`targeting="one_step"` on a held design. The fit projects the risk curve of each static arm onto a
working model and targets the coefficient scores of every cause together. The subsection "A
working model over the hazard curve" of
[hazard-based survival targeting](../longitudinal-tmle.md#hazard-based-survival-targeting)
states the estimand, the influence curve and the exact-law evidence.

The declared run passed every gated verdict, with no failed fit: every primary row and every
property cell with a verdict. The `iterated_mean_difference` rows are reported and carry no
verdict.

## The estimand and the laws

A cell is a static arm $r \in \{a_1, a_0\}$ and a visit $t = 1, \dots, T$, with weight
$h = 1$. The working model is $m(c, V; \beta) = \operatorname{expit}(\beta_0 + \beta_1 a_r +
\beta_2 t + \beta_3 a_r W_1)$ with $V = W_1$, and each cause $j$ has its own coefficients, the
solution of

$$
E\Bigl[\sum_c \varphi(c, V)\,\{F_j(t \mid a_r, W) - m(c, V; \beta)\}\Bigr] = 0,
\qquad \varphi(c, V) = (1, a_r, t, a_r W_1).
$$

The parameter is defined whether or not the working model holds, and it does not hold on these
laws. One calibration cell fits the same design under the identity link.

Every law is finite. $W_1 \in \{0, 1\}$ and $W_2 \in \{0, \dots, 3\}$ are independent and
uniform. The arm is logistic in $W$ and the visit dropout is logistic in the arm and $W$. At each
visit each cause occurs with a logistic pseudo-hazard in the visit, the arm, $W_1$, $W_2$ and
$a W_1$.

| law | visits | causes | $P(A = 1 \mid W)$ | what it carries |
| --- | --- | --- | --- | --- |
| `survival` | 5 | one | 0.15 to 0.76 when $W_1 = 0$, 0.45 when $W_1 = 1$ | the primary rows and every gated family |
| `competing` | 4 | relapse and death | as `survival` | the primary rows and calibration |
| `overlap` | 5 | one, the `survival` hazard | 0.055, 0.223, 0.587 and 0.875 at $W_2 = 0, \dots, 3$ when $W_1 = 0$, 0.206 when $W_1 = 1$ | the reported `iterated_mean_difference` family only |

The primary rows report `msm_regimen[(intercept)]`, `msm_regimen[a]`, `msm_regimen[t]` and
`msm_regimen[a_W1]` on `survival`, and the same four terms for each cause on `competing`, as
`msm_regimen[<term>, <cause>]`.

The gated laws have good overlap, and on them the hazard fit and the iterated-mean fit of the
same working model agree. With correctly specified parametric learners both fits are efficient,
so their difference shrinks faster than $n^{-1/2}$. The `overlap` law has the same hazard and a
smallest treatment probability of 0.055. On it the two fits differ by up to about 8% of an
efficiency standard error, and both are efficient, so they cannot differ by more than
$o_P(n^{-1/2})$ on a law where both nuisance fits are correct.

The laws are defined in `tests/studies/hazard_msm_law.py`. They are not the shared laws of
[point-treatment survival](point-treatment-survival.md). On the shared `survival` law the wrong
hazard learner's plug-in coefficient of $a W_1$ lies 0.07 efficiency standard errors from the
truth at n = 2,000, so a necessity cell there could not fail. The study's laws add the $a W_1$
term to each hazard and raise the $W_2$ coefficient.

Every truth is the exact projection of the law's cell risks, computed by a Newton solve that does
not call the estimator. Every efficiency bound is the exact standard deviation of
$D^*_\beta = M^{-1}\sum_c \varphi\,[D^{(1)}_c + F_j(t \mid a_r, W) - m(c, V; \beta)]$ over the
law's support. `tests/unit/test_hazard_msm_study_design.py` checks the truths against a root
finder and against a fit on each law's exact sample. It checks the bounds against the
cause-specific hazard form of the curve at every support point and against the influence curve
of a fit at n = 40,000.

## The fits

| nuisance | correct learner | wrong learner |
| --- | --- | --- |
| hazard | unpenalized logistic regression on `W1`, `W2`, the arm and the arm times `W1`, at each visit and cause | logistic regression on `W1` and the arm (`NoW2Logistic`) |
| treatment | logistic regression on `W1`, `W2` and `W1 W2` | the prior share |
| censoring | logistic regression on `W1`, `W2` and the arm, at each visit | the prior share |

The correct learners are correctly specified and are not saturated. With saturated treatment and
retention fits on a finite $W$, the targeted coefficients of the right and the wrong hazard
learner coincide, so no cell could see the hazard fit. The design test keeps that witness.

Every fit uses one fold, `tol=1e-10` and no simultaneous band, except the `survival` calibration
fits, which also report the simultaneous band.

No maintained package targets the coefficient scores of a working model over discrete-time
hazards, so no canonical implementation is compared. The exact truths and bounds, the property
families below and `tests/unit/test_hazard_msm.py` are the independent evidence.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| four visits, two competing causes, a binary baseline treatment, visit dropout; each cause's hazard has an a x W1 term and both mechanisms are logistic | `msm_regimen[(intercept), death]` | intercept of the logit working model (1, a, t, a W1) over the cumulative incidence of death by each visit under each static arm | `cleverly` | -0.0065 to 0.0042 | 0.9508 | 1.0081 | pass |
| four visits, two competing causes, a binary baseline treatment, visit dropout; each cause's hazard has an a x W1 term and both mechanisms are logistic | `msm_regimen[(intercept), relapse]` | intercept of the logit working model (1, a, t, a W1) over the cumulative incidence of relapse by each visit under each static arm | `cleverly` | -0.0092 to -0.000106 | 0.9542 | 1.0144 | pass |
| four visits, two competing causes, a binary baseline treatment, visit dropout; each cause's hazard has an a x W1 term and both mechanisms are logistic | `msm_regimen[a, death]` | treatment coefficient of the logit working model (1, a, t, a W1) over the cumulative incidence of death by each visit under each static arm | `cleverly` | -0.0182 to -0.000844 | 0.9546 | 0.9877 | pass |
| four visits, two competing causes, a binary baseline treatment, visit dropout; each cause's hazard has an a x W1 term and both mechanisms are logistic | `msm_regimen[a, relapse]` | treatment coefficient of the logit working model (1, a, t, a W1) over the cumulative incidence of relapse by each visit under each static arm | `cleverly` | -0.0103 to 0.0026 | 0.9558 | 1.0120 | pass |
| four visits, two competing causes, a binary baseline treatment, visit dropout; each cause's hazard has an a x W1 term and both mechanisms are logistic | `msm_regimen[a_W1, death]` | treatment-by-W1 coefficient of the logit working model (1, a, t, a W1) over the cumulative incidence of death by each visit under each static arm | `cleverly` | -0.0016 to 0.0160 | 0.9496 | 1.0075 | pass |
| four visits, two competing causes, a binary baseline treatment, visit dropout; each cause's hazard has an a x W1 term and both mechanisms are logistic | `msm_regimen[a_W1, relapse]` | treatment-by-W1 coefficient of the logit working model (1, a, t, a W1) over the cumulative incidence of relapse by each visit under each static arm | `cleverly` | -0.0136 to 0.0057 | 0.9496 | 0.9995 | pass |
| four visits, two competing causes, a binary baseline treatment, visit dropout; each cause's hazard has an a x W1 term and both mechanisms are logistic | `msm_regimen[t, death]` | visit coefficient of the logit working model (1, a, t, a W1) over the cumulative incidence of death by each visit under each static arm | `cleverly` | -0.000389 to 0.0019 | 0.9404 | 0.9663 | pass |
| four visits, two competing causes, a binary baseline treatment, visit dropout; each cause's hazard has an a x W1 term and both mechanisms are logistic | `msm_regimen[t, relapse]` | visit coefficient of the logit working model (1, a, t, a W1) over the cumulative incidence of relapse by each visit under each static arm | `cleverly` | -0.000322 to 0.0018 | 0.9496 | 1.0089 | pass |
| five visits, one cause, a binary baseline treatment, visit dropout; the hazard has an a x W1 term and the treatment and dropout mechanisms are logistic | `msm_regimen[(intercept)]` | intercept of the logit working model (1, a, t, a W1) over the risk by each visit under each static arm | `cleverly` | -0.0054 to 0.0023 | 0.9483 | 0.9874 | pass |
| five visits, one cause, a binary baseline treatment, visit dropout; the hazard has an a x W1 term and the treatment and dropout mechanisms are logistic | `msm_regimen[a]` | treatment coefficient of the logit working model (1, a, t, a W1) over the risk by each visit under each static arm | `cleverly` | -0.0035 to 0.0082 | 0.9542 | 1.0108 | pass |
| five visits, one cause, a binary baseline treatment, visit dropout; the hazard has an a x W1 term and the treatment and dropout mechanisms are logistic | `msm_regimen[a_W1]` | treatment-by-W1 coefficient of the logit working model (1, a, t, a W1) over the risk by each visit under each static arm | `cleverly` | -0.0141 to 0.0017 | 0.9529 | 1.0016 | pass |
| five visits, one cause, a binary baseline treatment, visit dropout; the hazard has an a x W1 term and the treatment and dropout mechanisms are logistic | `msm_regimen[t]` | visit coefficient of the logit working model (1, a, t, a W1) over the risk by each visit under each static arm | `cleverly` | -0.000960 to 0.000781 | 0.9533 | 0.9998 | pass |
<!-- /generated -->

The primary rows draw 2,400 samples of 2,000 rows for each scenario and report all twelve
coefficients. The shared primary rule (`repeated_sizing.primary_replicates`) sets the count from
the committed `survival_logit` and `competing_logit` calibration pilots.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `competing_logit_a_W1_death__mechanism_correct` | positive | the competing law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the cumulative incidence of death: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0073 to 0.0180, margin 0.0425, SE ratio 0.9968 | pass |
| `double_robustness` | `survival_logit_a_W1__both_correct` | positive | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0153 to 0.0073, margin 0.0379, SE ratio 0.9885 | pass |
| `double_robustness` | `survival_logit_a_W1__both_wrong` | control | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.3290 to -0.3074, margin 0.0363, SE ratio 0.9748 | pass |
| `double_robustness` | `survival_logit_a_W1__mechanism_correct` | positive | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0272 to -0.0050, margin 0.0373, SE ratio 1.0522 | pass |
| `double_robustness` | `survival_logit_a_W1__outcome_correct` | positive | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0091 to 0.0117, margin 0.0349, SE ratio 1.0066 | pass |
| `interval_calibration` | `competing_logit_a_W1_death__correctly_specified` | positive | the competing law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the cumulative incidence of death: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9370 to 0.9534, SE ratio 0.9660 to 1.0158, empirical efficiency ratio 0.9841 to 1.0351, reported efficiency ratio 0.9980 to 1.0017 | pass |
| `interval_calibration` | `competing_logit_a_W1_relapse__correctly_specified` | positive | the competing law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the cumulative incidence of relapse: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9398 to 0.9559, SE ratio 0.9668 to 1.0177, empirical efficiency ratio 0.9872 to 1.0393, reported efficiency ratio 1.0030 to 1.0062 | pass |
| `interval_calibration` | `competing_logit_a_death__correctly_specified` | positive | the competing law with an a x W1 hazard term, the treatment coefficient of the logit working model over the cumulative incidence of death: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9374 to 0.9537, SE ratio 0.9630 to 1.0126, empirical efficiency ratio 0.9872 to 1.0380, reported efficiency ratio 0.9978 to 1.0017 | pass |
| `interval_calibration` | `competing_logit_a_relapse__correctly_specified` | positive | the competing law with an a x W1 hazard term, the treatment coefficient of the logit working model over the cumulative incidence of relapse: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9425 to 0.9582, SE ratio 0.9794 to 1.0292, empirical efficiency ratio 0.9728 to 1.0219, reported efficiency ratio 0.9993 to 1.0024 | pass |
| `interval_calibration` | `competing_logit_intercept_death__correctly_specified` | positive | the competing law with an a x W1 hazard term, the intercept of the logit working model over the cumulative incidence of death: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9440 to 0.9594, SE ratio 0.9768 to 1.0284, empirical efficiency ratio 0.9747 to 1.0260, reported efficiency ratio 1.0012 to 1.0037 | pass |
| `interval_calibration` | `competing_logit_intercept_relapse__correctly_specified` | positive | the competing law with an a x W1 hazard term, the intercept of the logit working model over the cumulative incidence of relapse: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9365 to 0.9530, SE ratio 0.9565 to 1.0063, empirical efficiency ratio 0.9954 to 1.0475, reported efficiency ratio 1.0008 to 1.0032 | pass |
| `interval_calibration` | `competing_logit_t_death__correctly_specified` | positive | the competing law with an a x W1 hazard term, the visit coefficient of the logit working model over the cumulative incidence of death: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9460 to 0.9612, SE ratio 0.9859 to 1.0354, empirical efficiency ratio 0.9680 to 1.0167, reported efficiency ratio 1.0011 to 1.0036 | pass |
| `interval_calibration` | `competing_logit_t_relapse__correctly_specified` | positive | the competing law with an a x W1 hazard term, the visit coefficient of the logit working model over the cumulative incidence of relapse: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9427 to 0.9584, SE ratio 0.9750 to 1.0253, empirical efficiency ratio 0.9772 to 1.0276, reported efficiency ratio 1.0006 to 1.0033 | pass |
| `interval_calibration` | `survival_identity_a_W1__correctly_specified` | positive | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the identity-link working model over the risk curve: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9450 to 0.9603, SE ratio 0.9892 to 1.0413, empirical efficiency ratio 0.9603 to 1.0110, reported efficiency ratio 0.9983 to 1.0014 | pass |
| `interval_calibration` | `survival_logit_a_W1__correctly_specified` | positive | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9403 to 0.9562, SE ratio 0.9737 to 1.0236, empirical efficiency ratio 0.9794 to 1.0294, reported efficiency ratio 1.0015 to 1.0037 | pass |
| `interval_calibration` | `survival_logit_a_W1__noise_control` | control | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8233 to 0.8499, SE ratio 0.6983 to 0.7341, empirical efficiency ratio 1.3656 to 1.4354, reported efficiency ratio 1.0015 to 1.0038 | pass |
| `interval_calibration` | `survival_logit_a_W1__shrunken_se_control` | control | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8114 to 0.8387, SE ratio 0.6818 to 0.7174, empirical efficiency ratio 0.9783 to 1.0294, reported efficiency ratio 0.7011 to 0.7026 | pass |
| `interval_calibration` | `survival_logit_a__correctly_specified` | positive | the survival law with an a x W1 hazard term, the treatment coefficient of the logit working model over the risk curve: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9400 to 0.9560, SE ratio 0.9756 to 1.0266, empirical efficiency ratio 0.9742 to 1.0250, reported efficiency ratio 0.9986 to 1.0017 | pass |
| `interval_calibration` | `survival_logit_a__noise_control` | control | the survival law with an a x W1 hazard term, the treatment coefficient of the logit working model over the risk curve: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8106 to 0.8380, SE ratio 0.6824 to 0.7171, empirical efficiency ratio 1.3942 to 1.4651, reported efficiency ratio 0.9986 to 1.0017 | pass |
| `interval_calibration` | `survival_logit_a__shrunken_se_control` | control | the survival law with an a x W1 hazard term, the treatment coefficient of the logit working model over the risk curve: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8134 to 0.8406, SE ratio 0.6831 to 0.7178, empirical efficiency ratio 0.9754 to 1.0249, reported efficiency ratio 0.6989 to 0.7012 | pass |
| `interval_calibration` | `survival_logit_intercept__correctly_specified` | positive | the survival law with an a x W1 hazard term, the intercept of the logit working model over the risk curve: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9483 to 0.9631, SE ratio 0.9777 to 1.0279, empirical efficiency ratio 0.9739 to 1.0241, reported efficiency ratio 1.0003 to 1.0020 | pass |
| `interval_calibration` | `survival_logit_intercept__noise_control` | control | the survival law with an a x W1 hazard term, the intercept of the logit working model over the risk curve: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8154 to 0.8425, SE ratio 0.6856 to 0.7217, empirical efficiency ratio 1.3875 to 1.4600, reported efficiency ratio 1.0003 to 1.0020 | pass |
| `interval_calibration` | `survival_logit_intercept__shrunken_se_control` | control | the survival law with an a x W1 hazard term, the intercept of the logit working model over the risk curve: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8180 to 0.8449, SE ratio 0.6841 to 0.7195, empirical efficiency ratio 0.9741 to 1.0243, reported efficiency ratio 0.7002 to 0.7014 | pass |
| `interval_calibration` | `survival_logit_t__correctly_specified` | positive | the survival law with an a x W1 hazard term, the visit coefficient of the logit working model over the risk curve: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9407 to 0.9566, SE ratio 0.9635 to 1.0136, empirical efficiency ratio 0.9886 to 1.0396, reported efficiency ratio 1.0010 to 1.0029 | pass |
| `interval_calibration` | `survival_logit_t__noise_control` | control | the survival law with an a x W1 hazard term, the visit coefficient of the logit working model over the risk curve: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8182 to 0.8451, SE ratio 0.6771 to 0.7133, empirical efficiency ratio 1.4045 to 1.4796, reported efficiency ratio 1.0009 to 1.0029 | pass |
| `interval_calibration` | `survival_logit_t__shrunken_se_control` | control | the survival law with an a x W1 hazard term, the visit coefficient of the logit working model over the risk curve: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8071 to 0.8346, SE ratio 0.6740 to 0.7089, empirical efficiency ratio 0.9895 to 1.0405, reported efficiency ratio 0.7007 to 0.7020 | pass |
| `iterated_mean_difference` | `overlap_logit__iterated_curve_decrease` | diagnostic | the survival hazard under limited treatment overlap, the iterated-mean fit of the risk curve without a working model, n = 2,000: the largest decrease of the iterated-mean risk curve of either arm between consecutive visits | no verdict; the share of draws whose curve decreases somewhere is reported | decreasing share 0, 99% interval 0 to 0.0066 | reported |
| `iterated_mean_difference` | `overlap_logit_a_W1__hazard_fit` | diagnostic | the survival hazard under limited treatment overlap, the treatment-by-W1 coefficient of the logit working model over the risk curve: the hazard fit of the working model, against the exact coefficient | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9107 to 0.9565, SE ratio 0.9612 | reported |
| `iterated_mean_difference` | `overlap_logit_a_W1__iterated_fit` | diagnostic | the survival hazard under limited treatment overlap, the treatment-by-W1 coefficient of the logit working model over the risk curve: the iterated-mean fit of the same working model on the same draw, against the exact coefficient | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9136 to 0.9585, SE ratio 0.9627 | reported |
| `iterated_mean_difference` | `overlap_logit_a_W1__paired_difference` | diagnostic | the survival hazard under limited treatment overlap, the treatment-by-W1 coefficient of the logit working model over the risk curve: the hazard coefficient minus the iterated-mean coefficient of the same working model on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0524, mean difference -0.0015 to 0.000512 | reported |
| `iterated_mean_difference` | `overlap_logit_a__hazard_fit` | diagnostic | the survival hazard under limited treatment overlap, the treatment coefficient of the logit working model over the risk curve: the hazard fit of the working model, against the exact coefficient | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9078 to 0.9544, SE ratio 0.9675 | reported |
| `iterated_mean_difference` | `overlap_logit_a__iterated_fit` | diagnostic | the survival hazard under limited treatment overlap, the treatment coefficient of the logit working model over the risk curve: the iterated-mean fit of the same working model on the same draw, against the exact coefficient | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9165 to 0.9606, SE ratio 0.9772 | reported |
| `iterated_mean_difference` | `overlap_logit_a__paired_difference` | diagnostic | the survival hazard under limited treatment overlap, the treatment coefficient of the logit working model over the risk curve: the hazard coefficient minus the iterated-mean coefficient of the same working model on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0738, mean difference -0.000561 to 0.0014 | reported |
| `iterated_mean_difference` | `overlap_logit_intercept__hazard_fit` | diagnostic | the survival hazard under limited treatment overlap, the intercept of the logit working model over the risk curve: the hazard fit of the working model, against the exact coefficient | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9238 to 0.9657, SE ratio 1.0245 | reported |
| `iterated_mean_difference` | `overlap_logit_intercept__iterated_fit` | diagnostic | the survival hazard under limited treatment overlap, the intercept of the logit working model over the risk curve: the iterated-mean fit of the same working model on the same draw, against the exact coefficient | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9267 to 0.9677, SE ratio 1.0274 | reported |
| `iterated_mean_difference` | `overlap_logit_intercept__paired_difference` | diagnostic | the survival hazard under limited treatment overlap, the intercept of the logit working model over the risk curve: the hazard coefficient minus the iterated-mean coefficient of the same working model on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0272, mean difference -0.000185 to 0.000220 | reported |
| `iterated_mean_difference` | `overlap_logit_t__hazard_fit` | diagnostic | the survival hazard under limited treatment overlap, the visit coefficient of the logit working model over the risk curve: the hazard fit of the working model, against the exact coefficient | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9462 to 0.9805, SE ratio 1.0373 | reported |
| `iterated_mean_difference` | `overlap_logit_t__iterated_fit` | diagnostic | the survival hazard under limited treatment overlap, the visit coefficient of the logit working model over the risk curve: the iterated-mean fit of the same working model on the same draw, against the exact coefficient | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9462 to 0.9805, SE ratio 1.0390 | reported |
| `iterated_mean_difference` | `overlap_logit_t__paired_difference` | diagnostic | the survival hazard under limited treatment overlap, the visit coefficient of the logit working model over the risk curve: the hazard coefficient minus the iterated-mean coefficient of the same working model on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0328, mean difference -0.000048 to 0.000072 | reported |
| `iterated_mean_difference` | `survival_logit_a_W1__mechanism_correct_difference` | diagnostic | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: the hazard coefficient minus the iterated-mean coefficient of the same working model on the same draw, with both outcome regressions omitting W2 and the treatment and censoring models correct | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0433, mean difference -0.0011 to -0.000183 | reported |
| `iterated_mean_difference` | `survival_logit_a__mechanism_correct_difference` | diagnostic | the survival law with an a x W1 hazard term, the treatment coefficient of the logit working model over the risk curve: the hazard coefficient minus the iterated-mean coefficient of the same working model on the same draw, with both outcome regressions omitting W2 and the treatment and censoring models correct | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0748, mean difference 0.000272 to 0.0015 | reported |
| `iterated_mean_difference` | `survival_logit_intercept__mechanism_correct_difference` | diagnostic | the survival law with an a x W1 hazard term, the intercept of the logit working model over the risk curve: the hazard coefficient minus the iterated-mean coefficient of the same working model on the same draw, with both outcome regressions omitting W2 and the treatment and censoring models correct | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0394, mean difference -0.000324 to 0.000095 | reported |
| `iterated_mean_difference` | `survival_logit_t__mechanism_correct_difference` | diagnostic | the survival law with an a x W1 hazard term, the visit coefficient of the logit working model over the risk curve: the hazard coefficient minus the iterated-mean coefficient of the same working model on the same draw, with both outcome regressions omitting W2 and the treatment and censoring models correct | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0399, mean difference -0.000069 to 0.000029 | reported |
| `power` | `survival_logit_a_W1__alternative` | positive | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `survival_logit_a_W1__n_1000` | control | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias 0.000441, coverage 0.9165 to 0.9606, SE ratio 0.9860 | pass |
| `root_n_and_efficiency` | `survival_logit_a_W1__n_2000` | positive | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0032, coverage 0.9282 to 0.9688, SE ratio 1.0078 | pass |
| `root_n_and_efficiency` | `survival_logit_a_W1__n_8000` | positive | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0016, coverage 0.9223 to 0.9647, SE ratio 0.9630 | pass |
| `root_n_and_efficiency` | `survival_logit_a__n_1000` | control | the survival law with an a x W1 hazard term, the treatment coefficient of the logit working model over the risk curve: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0089, coverage 0.9238 to 0.9657, SE ratio 0.9955 | pass |
| `root_n_and_efficiency` | `survival_logit_a__n_2000` | positive | the survival law with an a x W1 hazard term, the treatment coefficient of the logit working model over the risk curve: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.0023, coverage 0.9341 to 0.9727, SE ratio 0.9896 | pass |
| `root_n_and_efficiency` | `survival_logit_a__n_8000` | positive | the survival law with an a x W1 hazard term, the treatment coefficient of the logit working model over the risk curve: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000284, coverage 0.9282 to 0.9688, SE ratio 0.9909 | pass |
| `root_n_and_efficiency` | `survival_logit_intercept__n_1000` | control | the survival law with an a x W1 hazard term, the intercept of the logit working model over the risk curve: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.000765, coverage 0.9165 to 0.9606, SE ratio 0.9598 | pass |
| `root_n_and_efficiency` | `survival_logit_intercept__n_2000` | positive | the survival law with an a x W1 hazard term, the intercept of the logit working model over the risk curve: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0054, coverage 0.9238 to 0.9657, SE ratio 1.0140 | pass |
| `root_n_and_efficiency` | `survival_logit_intercept__n_8000` | positive | the survival law with an a x W1 hazard term, the intercept of the logit working model over the risk curve: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000697, coverage 0.9311 to 0.9708, SE ratio 1.0015 | pass |
| `root_n_and_efficiency` | `survival_logit_t__n_1000` | control | the survival law with an a x W1 hazard term, the visit coefficient of the logit working model over the risk curve: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias 0.000234, coverage 0.9020 to 0.9502, SE ratio 0.9331 | pass |
| `root_n_and_efficiency` | `survival_logit_t__n_2000` | positive | the survival law with an a x W1 hazard term, the visit coefficient of the logit working model over the risk curve: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000923, coverage 0.9208 to 0.9637, SE ratio 0.9723 | pass |
| `root_n_and_efficiency` | `survival_logit_t__n_8000` | positive | the survival law with an a x W1 hazard term, the visit coefficient of the logit working model over the risk curve: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000095, coverage 0.9179 to 0.9616, SE ratio 0.9759 | pass |
| `root_n_rate` | `survival_logit_a_W1__empirical_sd` | positive | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5289 to -0.4463 | pass |
| `root_n_rate` | `survival_logit_a_W1__reported_se` | positive | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5037 to -0.4999 | pass |
| `root_n_rate` | `survival_logit_a__empirical_sd` | positive | the survival law with an a x W1 hazard term, the treatment coefficient of the logit working model over the risk curve: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5419 to -0.4580 | pass |
| `root_n_rate` | `survival_logit_a__reported_se` | positive | the survival law with an a x W1 hazard term, the treatment coefficient of the logit working model over the risk curve: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5036 to -0.4982 | pass |
| `root_n_rate` | `survival_logit_intercept__empirical_sd` | positive | the survival law with an a x W1 hazard term, the intercept of the logit working model over the risk curve: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5601 to -0.4751 | pass |
| `root_n_rate` | `survival_logit_intercept__reported_se` | positive | the survival law with an a x W1 hazard term, the intercept of the logit working model over the risk curve: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5022 to -0.4993 | pass |
| `root_n_rate` | `survival_logit_t__empirical_sd` | positive | the survival law with an a x W1 hazard term, the visit coefficient of the logit working model over the risk curve: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5632 to -0.4768 | pass |
| `root_n_rate` | `survival_logit_t__reported_se` | positive | the survival law with an a x W1 hazard term, the visit coefficient of the logit working model over the risk curve: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5030 to -0.4995 | pass |
| `simultaneous_coverage` | `survival_logit_coefficients__pointwise_joint_control` | control | the survival law with an a x W1 hazard term, the four coefficients of the logit working model read as one joint set: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8235 to 0.8501 | pass |
| `simultaneous_coverage` | `survival_logit_coefficients__simultaneous_band` | positive | the survival law with an a x W1 hazard term, the four coefficients of the logit working model read as one joint set: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9370 to 0.9534 | pass |
| `targeting_necessity` | `survival_logit_a_W1__targeted` | positive | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0171 to 0.0059, margin 0.0386 | pass |
| `targeting_necessity` | `survival_logit_a_W1__untargeted` | control | the survival law with an a x W1 hazard term, the treatment-by-W1 coefficient of the logit working model over the risk curve: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.8713 to 0.8836, margin 0.0206 | pass |
<!-- /generated -->

| family | cells | replicates and n | what fails it |
| --- | --- | --- | --- |
| `interval_calibration` | each coefficient of the three fits, with the shrunken-SE and noise controls on the four `survival` coefficients | 5,200 at n = 2,000 | an SE ratio or coverage interval outside its band, or an efficiency ratio outside 0.90 to 1.10 |
| `simultaneous_coverage` | the band over the four `survival` coefficients, and its pointwise control | 5,200 at n = 2,000 | joint coverage outside the calibration band |
| `double_robustness` | the `survival` coefficient of $a W_1$ with both, one or neither nuisance correct, and the `competing` death coefficient of $a W_1$ with the wrong hazard learner and the correct mechanism | 1,200 at n = 2,000 | a positive bias interval outside the margin, or a `both_wrong` interval inside it |
| `targeting_necessity` | the same coefficient from the wrong hazard learner, targeted and plug-in | 1,200 at n = 2,000 | the targeted bias outside the margin, or a displacement below 0.25 |
| `root_n_and_efficiency`, `root_n_rate` | the four `survival` coefficients at n = 1,000, 2,000 and 8,000; the n = 1,000 rung is a rate rung that carries no claim | 800 at each n | the shared root-n rules at n = 2,000 and 8,000, and the rate across all three |
| `power` | the $a W_1$ coefficient against zero | 800 at n = 2,000 | a rejection lower bound below the minimum power |
| `iterated_mean_difference` | on the `overlap` law: the hazard coefficient minus the iterated-mean coefficient of the same draw, each fit against the truth, and the largest decrease of the iterated-mean risk curve. On the `double_robustness` `mechanism_correct` draws: the same paired difference with both regressions misspecified | 800 at n = 2,000, and the 1,200 robustness draws | nothing: the family is reported with no verdict |

The calibration count is the arm-pooled SE-ratio rule of `mtp_study_tools.smallest_replicates`:
each cell passes with probability at least 0.90, and all thirteen arms pass together with
probability at least 0.95, read from the committed calibration pilot of 8,000 draws per fit. Its
coverage runs from 0.947 to 0.954 and its SE ratio from 0.99 to 1.02 on every coefficient. At 5,200
replicates the shrunken-SE control
leaves the SE-ratio band and the noise control leaves the efficiency band with probability above
0.999.

The wrong hazard learner's plug-in coefficient of $a W_1$ has an exact limit 5.9 efficiency
standard errors from the truth at n = 2,000, so the plug-in arm of `targeting_necessity` can fail.
With both nuisances wrong the exact limit lies 2.1 efficiency standard errors from the truth, so
the `both_wrong` control can fail; if it does not clear the margin, the control has failed and
the study is a defect suspect. On `competing` the wrong hazard learner's plug-in death
coefficient of $a W_1$ lies 4.6 efficiency standard errors from the truth, while its targeted
limit is the truth, so that cell reads the clever covariate that carries the relapse hazard into
the death update. The power cell's exact-law power at n = 2,000 is above 0.99.

At bias zero every positive bias arm passes with probability above 0.99. That probability stays
above 0.99 while the true bias is below 0.108 standard deviations of the arm at 1,200 replicates
and below 0.077 at 800. The 200-draw wrong-learner pilot reads a targeted bias of 0.005 standard
deviations for $a W_1$, and the 8,000-draw calibration pilots read the `both_correct` arm and the
n = 2,000 rungs.

The `iterated_mean_difference` rows read the `overlap` law. They measure how far the hazard fit
and the iterated-mean fit of the same working model sit apart, how each fit's interval
calibrates, and how often the iterated-mean risk curve, fitted without a working model and
without isotonic repair, decreases between visits. The summary reports, for each coefficient,
the SD of the paired difference over the efficiency-bound standard error and each fit's coverage
and SE ratio side by side. It also reports the share of decreasing curves with its exact
interval. These rows describe the two constructions; they are not a claim about either.

Before the run, the design test requires the committed 200-draw pilot to reach a
paired-difference ratio of at least 0.05 on the largest coefficient and on $a W_1$. The pilot
reads these ratios:

| coefficient | paired ratio, `overlap` law | paired ratio, `mechanism_correct` draws |
| --- | --- | --- |
| intercept | 0.028 | 0.038 |
| $a$ | 0.077 | 0.070 |
| $t$ | 0.035 | 0.038 |
| $a W_1$ | 0.054 | 0.043 |

So the two fits separate by 3 to 8% of an efficiency standard error on the `overlap` law, and by
4 to 7% when both regressions omit $W_2$ and the mechanism is correct. Neither reading separates
them at first order. A Wald standard-error shortfall on the `overlap` law is shared by the two
fits and small: the 200-draw pilot reads an SE ratio of 0.92 for the intercept in both fits, and
an 800-draw planning run read 0.94 (hazard) and 0.95 (iterated mean) for the arm coefficient. It
is reported and not gated. The share of decreasing curves carries no floor: the pilot found none
in 200 draws, with a 99% interval of 0 to 0.026.

The declared run reads the same picture on 800 `overlap` draws and on the 1,200 `mechanism_correct`
draws. The paired ratio of $a W_1$ is 0.0524 on the `overlap` law and 0.0433 on the
`mechanism_correct` draws, and the largest is 0.0748, for $a$ on the `mechanism_correct` draws. On
the `overlap` law the two fits read SE ratios of 0.9612 and 0.9627 for $a W_1$, and no
iterated-mean curve decreased in 800 draws.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications |
| `n` | 2000 | observations per replication of the record's reference size |
| `independent_tests_passed` | 12 | truth tests passing |
| `independent_tests_total` | 12 | truth tests reported |
| `property_cells_passed` | 51 | property cells passing |
| `property_cells_total` | 51 | property cells reported |
| `max_standardized_bias` | 0.0578 | largest primary standardized bias |
| `min_coverage` | 0.9404 | lowest primary coverage |
| `properties[interval_calibration/survival_logit_a_W1__correctly_specified]:se_ratio` | 0.9988 | SE ratio of the calibration cell of $a W_1$ |
| `properties[interval_calibration/survival_logit_a_W1__correctly_specified]:coverage` | 0.9487 | coverage of the calibration cell of $a W_1$ |
| `properties[double_robustness/survival_logit_a_W1__both_wrong]:standardized_bias` | -2.1943 | standardized bias of the both-wrong control |
| `properties[double_robustness/competing_logit_a_W1_death__mechanism_correct]:standardized_bias` | 0.0313 | standardized bias of the cross-cause cell |
| `properties[targeting_necessity/survival_logit_a_W1__untargeted]:standardized_bias` | 10.6422 | standardized bias of the plug-in control |
| `properties[simultaneous_coverage/survival_logit_coefficients__simultaneous_band]:coverage` | 0.9456 | joint coverage of the band |
| `properties[iterated_mean_difference/overlap_logit_a_W1__paired_difference]:paired_sd_over_bound` | 0.0524 | paired ratio of $a W_1$, `overlap` law |
| `properties[iterated_mean_difference/survival_logit_a_W1__mechanism_correct_difference]:paired_sd_over_bound` | 0.0433 | paired ratio of $a W_1$, `mechanism_correct` draws |
| `properties[iterated_mean_difference/survival_logit_a__mechanism_correct_difference]:paired_sd_over_bound` | 0.0748 | paired ratio of $a$, `mechanism_correct` draws |
| `properties[iterated_mean_difference/overlap_logit_a_W1__hazard_fit]:se_ratio` | 0.9612 | SE ratio of the hazard fit of $a W_1$, `overlap` law |
| `properties[iterated_mean_difference/overlap_logit_a_W1__iterated_fit]:se_ratio` | 0.9627 | SE ratio of the iterated-mean fit of $a W_1$, `overlap` law |
| `properties[iterated_mean_difference/overlap_logit__iterated_curve_decrease]:rejection_rate` | 0 | share of decreasing iterated-mean curves |
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
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 24 | the same bound, in primary replications |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |

## Limitations

| limit | what it means for use |
| --- | --- |
| one-step targeting with a working model only | the per-horizon algorithm and `isotonic=True` are refused beside `msm=`, and the cell fits without `msm=` are covered by [point survival hazard](point-survival-hazard.md) |
| finite laws with discrete covariates | the evidence covers correct or misspecified low-dimensional logistic learners. A flexible learner is outside it |
| uniform cell weights and $V = W_1$ | a projection weight that varies with $V$ or the cell is covered by the exact-law tests only |
| no comparator | no external implementation targets these coefficient scores |
| no cross-fitting, observation weights or clusters | each is a separate composition and needs its own row |

## Reproduction

Run `python tests/canonical/hazard_msm/regenerate.py --output DIR` with an empty scratch
directory outside the repository to regenerate the artifacts. The pilot runs separately, as
`python -m tests.studies.hazard_msm_pilot --jobs N`, and writes
`tests/canonical/hazard_msm/pilot.csv.gz`. The red-cell readings are declared in
`tests/canonical/hazard_msm/regenerate.py` before any run.
