# Stratified point-treatment TMLE

This study validates ordinary point-treatment TMLE with baseline strata, `strata=["V"]`. A stratum
parameter is the marginal point-treatment parameter of the law inside one of a fixed number of
baseline strata. The construction is the marginal result applied inside each cell of a finite
partition, which is a standard step under the roadmap's
[Eligibility](../../roadmap.md#eligibility) conditions.

The fit fluctuates once. Its submodel carries one disjoint score block
$I(V=s) H_s / P_n(V=s)$ for each stratum $s$. $H_s$ is the marginal clever covariate built with
the within-stratum arm shares. Each stratum curve is $I(V=s) D_s / P_n(V=s)$, where $D_s$ is
the marginal influence curve computed inside the stratum. The fit also reports the marginal
parameters, and the marginal estimate is the empirical mixture of the stratum estimates.

## What was tested

| setting | declaration |
| --- | --- |
| stratum | $V \in \{0, 1, 2\}$ with probabilities 0.5, 0.3 and 0.2. The strata are unequal, so the $n / n_s$ weight of a stratum curve matters |
| covariate | $W \in \{0, 1, 2\}$ given $V$, with rows $(0.5, 0.3, 0.2)$, $(0.3, 0.4, 0.3)$ and $(0.2, 0.3, 0.5)$ |
| treatment | $g(w, v) = \operatorname{expit}(-0.3 + 0.5 w - 0.4 v)$, between 0.25 and 0.67 |
| outcome | binary, $Q(a, w, v) = \operatorname{expit}(-0.8 + a + 0.6 w + 0.9 v)$ |
| truth | exact. Every variable takes finitely many values, so `tests/studies/stratified_law.py` sums each truth and each efficient influence function over the 36 support points |
| fit | `TMLE(cross_fit=False, estimands=("ey", "ate"))` with main-terms logistic regressions on $(A, W, V)$ and $(W, V)$, which are correct for this law, and `g_bounds=(0.01, 0.99)` |
| primary scenario | `stratified_binary`, the two arm means and the ATE in each stratum, $n = 2000$, 1,000 replications |
| Monte Carlo inference | exact 99% intervals around every rate, and 99% intervals around every primary endpoint |

The law module records why two coefficients of $Q$ moved before the declaration commit. At the
first declared values the stratum ATEs were too close together, and the marginal-fluctuation
control could not move 1 SD in stratum 0. The change also raised the treated mean of stratum 2
from 0.858 to 0.936.

The comparator is R `tmle3` at commit `ed72f8a`, through `tmle_stratified(..., base_estimate = FALSE)`
in two stages. The first stage is `tmle_TSM_all()` and gives the six stratum arm means. The second
stage is `tmle_ATE(1, 0)` and gives the three stratum ATEs. `base_estimate = FALSE` drops the
marginal parameter, because the marginal clever covariate is the sum of the stratum columns and
would make the update collinear. [References](../../references.md#targeted-learning-in-general) gives the source locators.

The two implementations differ in one declared way. The `tmle3` ATE stage fluctuates one
clever-covariate column per stratum. The `cleverly` fit with `estimands=("ey", "ate")`
fluctuates the two arm columns of each stratum, as the
[ordinary point-treatment study](canonical-point-treatment-tmle.md) pairs it.

Two facts about the comparator shape the runner, `tests/canonical/tmle3_stratified/run_tmle3_stratified.R`.

| fact | what the runner does |
| --- | --- |
| `fit$summary` fails on a stratified spec with two base parameters. The fit stores the parameter names and initial estimates as matrices, and the summary builds 12 columns and names them with 10 names | it computes the same summary from `fit$estimates`: the covariance of the stacked curves with an $n - 1$ denominator, $\sqrt{\operatorname{diag} / n}$, and the Wald interval at 1.959964 |
| `get_strata_weights` merges the stratum column with the strata table under `sort = FALSE` | it checks on every replication that each row's one nonzero weight column is its own stratum, and that the weight is $n / n_s$ |

The table gives the estimands that are measured against the exact truth and not paired.

| estimand | why it is not paired |
| --- | --- |
| ATT and ATC | `Param_ATT$estimates` and `Param_ATC$estimates` read the counterfactual tasks of the training task, not of the task they receive. On a stratum subset the arm means and the treated share are then full-length vectors beside a subset-length outcome, and the estimate is the marginal one. The two parameters also update the treatment node, and `cleverly` does not fluctuate it |
| PAR | `Param_PAR` is not verified as safe on a subset at `ed72f8a` |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law with three unequal baseline strata | `ate[V=0]` | average treatment effect, in stratum V=0 | `cleverly` ordinary TMLE with baseline strata | -0.0026 to 0.0024 | 0.9500 | 1.0083 | pass |
| binary-outcome law with three unequal baseline strata | `ate[V=0]` | average treatment effect, in stratum V=0 | R `tmle3` stratified TMLE, `tmle_stratified` | -0.0026 to 0.0024 | 0.9500 | 1.0085 | pass |
| binary-outcome law with three unequal baseline strata | `ate[V=1]` | average treatment effect, in stratum V=1 | `cleverly` ordinary TMLE with baseline strata | -0.0014 to 0.0043 | 0.9430 | 0.9957 | pass |
| binary-outcome law with three unequal baseline strata | `ate[V=1]` | average treatment effect, in stratum V=1 | R `tmle3` stratified TMLE, `tmle_stratified` | -0.0014 to 0.0043 | 0.9440 | 0.9977 | pass |
| binary-outcome law with three unequal baseline strata | `ate[V=2]` | average treatment effect, in stratum V=2 | `cleverly` ordinary TMLE with baseline strata | -0.0044 to 0.000590 | 0.9510 | 1.0001 | pass |
| binary-outcome law with three unequal baseline strata | `ate[V=2]` | average treatment effect, in stratum V=2 | R `tmle3` stratified TMLE, `tmle_stratified` | -0.0044 to 0.000587 | 0.9500 | 1.0018 | pass |
| binary-outcome law with three unequal baseline strata | `ey[0][V=0]` | counterfactual mean under treatment arm '0', in stratum V=0 | `cleverly` ordinary TMLE with baseline strata | -0.0030 to 0.000807 | 0.9420 | 0.9674 | pass |
| binary-outcome law with three unequal baseline strata | `ey[0][V=0]` | counterfactual mean under treatment arm '0', in stratum V=0 | R `tmle3` stratified TMLE, `tmle_stratified` | -0.0030 to 0.000807 | 0.9420 | 0.9674 | pass |
| binary-outcome law with three unequal baseline strata | `ey[0][V=1]` | counterfactual mean under treatment arm '0', in stratum V=1 | `cleverly` ordinary TMLE with baseline strata | -0.0027 to 0.0017 | 0.9480 | 0.9747 | pass |
| binary-outcome law with three unequal baseline strata | `ey[0][V=1]` | counterfactual mean under treatment arm '0', in stratum V=1 | R `tmle3` stratified TMLE, `tmle_stratified` | -0.0027 to 0.0017 | 0.9480 | 0.9747 | pass |
| binary-outcome law with three unequal baseline strata | `ey[0][V=2]` | counterfactual mean under treatment arm '0', in stratum V=2 | `cleverly` ordinary TMLE with baseline strata | -0.000833 to 0.0028 | 0.9430 | 1.0058 | pass |
| binary-outcome law with three unequal baseline strata | `ey[0][V=2]` | counterfactual mean under treatment arm '0', in stratum V=2 | R `tmle3` stratified TMLE, `tmle_stratified` | -0.000833 to 0.0028 | 0.9430 | 1.0058 | pass |
| binary-outcome law with three unequal baseline strata | `ey[1][V=0]` | counterfactual mean under treatment arm '1', in stratum V=0 | `cleverly` ordinary TMLE with baseline strata | -0.0030 to 0.000506 | 0.9570 | 1.0167 | pass |
| binary-outcome law with three unequal baseline strata | `ey[1][V=0]` | counterfactual mean under treatment arm '1', in stratum V=0 | R `tmle3` stratified TMLE, `tmle_stratified` | -0.0030 to 0.000506 | 0.9570 | 1.0167 | pass |
| binary-outcome law with three unequal baseline strata | `ey[1][V=1]` | counterfactual mean under treatment arm '1', in stratum V=1 | `cleverly` ordinary TMLE with baseline strata | -0.000962 to 0.0028 | 0.9470 | 1.0195 | pass |
| binary-outcome law with three unequal baseline strata | `ey[1][V=1]` | counterfactual mean under treatment arm '1', in stratum V=1 | R `tmle3` stratified TMLE, `tmle_stratified` | -0.000962 to 0.0028 | 0.9470 | 1.0194 | pass |
| binary-outcome law with three unequal baseline strata | `ey[1][V=2]` | counterfactual mean under treatment arm '1', in stratum V=2 | `cleverly` ordinary TMLE with baseline strata | -0.0027 to 0.000813 | 0.9170 | 0.9850 | **fail** |
| binary-outcome law with three unequal baseline strata | `ey[1][V=2]` | counterfactual mean under treatment arm '1', in stratum V=2 | R `tmle3` stratified TMLE, `tmle_stratified` | -0.0027 to 0.000813 | 0.9170 | 0.9849 | **fail** |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law with three unequal baseline strata | `ate[V=0]` | average treatment effect, in stratum V=0 | 0.000001 | 0.000282 | 1.0004 | 0 | 0.000713 vs 0.0500 | equivalent |
| binary-outcome law with three unequal baseline strata | `ate[V=1]` | average treatment effect, in stratum V=1 | 0.000007 | 0.0014 | 1.0037 | -0.0010 | 0.0021 vs 0.0500 | equivalent |
| binary-outcome law with three unequal baseline strata | `ate[V=2]` | average treatment effect, in stratum V=2 | 0.000001 | 0.000256 | 1.0031 | 0.0010 | 0.0055 vs 0.0500 | equivalent |
| binary-outcome law with three unequal baseline strata | `ey[0][V=0]` | counterfactual mean under treatment arm '0', in stratum V=0 | 4.362e-08 | 0.000012 | 1.0000 | 0 | 0.000017 vs 0.0500 | equivalent |
| binary-outcome law with three unequal baseline strata | `ey[0][V=1]` | counterfactual mean under treatment arm '0', in stratum V=1 | 4.438e-08 | 0.000011 | 1.0000 | 0 | 0.000008 vs 0.0500 | equivalent |
| binary-outcome law with three unequal baseline strata | `ey[0][V=2]` | counterfactual mean under treatment arm '0', in stratum V=2 | -1.943e-08 | 0.000006 | 1.0000 | 0 | 0.000076 vs 0.0500 | equivalent |
| binary-outcome law with three unequal baseline strata | `ey[1][V=0]` | counterfactual mean under treatment arm '1', in stratum V=0 | 3.601e-09 | 0.000001 | 1.0000 | 0 | 0.000018 vs 0.0500 | equivalent |
| binary-outcome law with three unequal baseline strata | `ey[1][V=1]` | counterfactual mean under treatment arm '1', in stratum V=1 | 4.951e-08 | 0.000014 | 1.0000 | 0 | 0.000010 vs 0.0500 | equivalent |
| binary-outcome law with three unequal baseline strata | `ey[1][V=2]` | counterfactual mean under treatment arm '1', in stratum V=2 | 1.511e-07 | 0.000047 | 1.0000 | 0 | 0.000140 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

The property cells fit only `cleverly`, against the exact truth. A label `v<s>_<parameter>` names
one stratum parameter.

| family | cells | n | replications |
| --- | --- | ---: | ---: |
| `interval_calibration` | each of the 18 stratum parameters of in-sample fits with `("ey", "ate", "att", "atc", "par")`, with a shrunken-SE and a noise control each | 2,000 | 2,400 |
| `interval_calibration` | the three stratum ATEs of the default cross-fitted fit, `cross_fit=True` with ten folds, with the same controls | 2,000 | 2,400 |
| `simultaneous_coverage` | `strata`, the default band over the 24 parameters of the in-sample fits, and `crossfit_strata`, the band over the 12 parameters of the cross-fitted fits | 2,000 | 2,400 |
| `double_robustness` | each stratum ATE with both nuisances correct, one correct, or neither. A wrong model is intercept-only | 2,000 | 1,200 |
| `stratum_targeting_necessity` | the outer stratum ATEs of a fit whose outcome regression omits $V$, against a longhand marginal fluctuation of the same regression on the same draws | 2,000 | 1,200 |

The marginal fluctuation solves the marginal score only. A regression that omits $V$ therefore
stays biased inside a stratum. The stratified fit solves each stratum's score, and it is
consistent there because $g$ is correct. The bias of the control is close to linear in $V$ and
its average is zero, so it nearly vanishes in the middle stratum. The family reads the two outer
strata, where `tests/unit/test_simultaneous_cell_design.py` puts the control at least 1 SD from
the truth. `tests/unit/test_stratified_influence_exact.py` checks the score block of every stratum.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `v0_ate__both_correct` | positive | average treatment effect in stratum V = 0: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0015 to 0.0031, margin 0.0077, SE ratio 1.0051 | pass |
| `double_robustness` | `v0_ate__both_wrong` | control | average treatment effect in stratum V = 0: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0383 to 0.0430, margin 0.0078, SE ratio 0.9769 | pass |
| `double_robustness` | `v0_ate__outcome_correct` | positive | average treatment effect in stratum V = 0: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0030 to 0.0014, margin 0.0074, SE ratio 1.0047 | pass |
| `double_robustness` | `v0_ate__treatment_correct` | positive | average treatment effect in stratum V = 0: only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0023 to 0.0024, margin 0.0079, SE ratio 1.0156 | pass |
| `double_robustness` | `v1_ate__both_correct` | positive | average treatment effect in stratum V = 1: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0024 to 0.0028, margin 0.0087, SE ratio 0.9927 | pass |
| `double_robustness` | `v1_ate__both_wrong` | control | average treatment effect in stratum V = 1: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0272 to 0.0322, margin 0.0083, SE ratio 1.0316 | pass |
| `double_robustness` | `v1_ate__outcome_correct` | positive | average treatment effect in stratum V = 1: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0034 to 0.0018, margin 0.0086, SE ratio 0.9770 | pass |
| `double_robustness` | `v1_ate__treatment_correct` | positive | average treatment effect in stratum V = 1: only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0031 to 0.0020, margin 0.0085, SE ratio 1.0482 | pass |
| `double_robustness` | `v2_ate__both_correct` | positive | average treatment effect in stratum V = 2: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0037 to 0.000930, margin 0.0078, SE ratio 0.9859 | pass |
| `double_robustness` | `v2_ate__both_wrong` | control | average treatment effect in stratum V = 2: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0141 to 0.0186, margin 0.0075, SE ratio 1.0358 | pass |
| `double_robustness` | `v2_ate__outcome_correct` | positive | average treatment effect in stratum V = 2: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0029 to 0.0016, margin 0.0076, SE ratio 1.0211 | pass |
| `double_robustness` | `v2_ate__treatment_correct` | positive | average treatment effect in stratum V = 2: only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0021 to 0.0027, margin 0.0080, SE ratio 0.9806 | pass |
| `interval_calibration` | `v0_atc__correctly_specified` | positive | average effect on the untreated in stratum V = 0: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9388 to 0.9619, SE ratio 0.9537 to 1.0259, empirical efficiency ratio 0.9752 to 1.0490, reported efficiency ratio 0.9994 to 1.0015 | pass |
| `interval_calibration` | `v0_atc__noise_control` | control | average effect on the untreated in stratum V = 0: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8155 to 0.8548, SE ratio 0.6833 to 0.7384, empirical efficiency ratio 1.3549 to 1.4640, reported efficiency ratio 0.9995 to 1.0015 | pass |
| `interval_calibration` | `v0_atc__shrunken_se_control` | control | average effect on the untreated in stratum V = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7994 to 0.8402, SE ratio 0.6678 to 0.7193, empirical efficiency ratio 0.9737 to 1.0485, reported efficiency ratio 0.6996 to 0.7011 | pass |
| `interval_calibration` | `v0_ate__correctly_specified` | positive | average treatment effect in stratum V = 0: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9388 to 0.9619, SE ratio 0.9492 to 1.0210, empirical efficiency ratio 0.9790 to 1.0534, reported efficiency ratio 0.9989 to 1.0006 | pass |
| `interval_calibration` | `v0_ate__noise_control` | control | average treatment effect in stratum V = 0: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8094 to 0.8493, SE ratio 0.6862 to 0.7385, empirical efficiency ratio 1.3539 to 1.4572, reported efficiency ratio 0.9989 to 1.0006 | pass |
| `interval_calibration` | `v0_ate__shrunken_se_control` | control | average treatment effect in stratum V = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7951 to 0.8362, SE ratio 0.6639 to 0.7152, empirical efficiency ratio 0.9783 to 1.0540, reported efficiency ratio 0.6992 to 0.7004 | pass |
| `interval_calibration` | `v0_ate_crossfit__correctly_specified` | positive | average treatment effect in stratum V = 0, from the cross-fitted fit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9397 to 0.9626, SE ratio 0.9567 to 1.0288, empirical efficiency ratio 0.9757 to 1.0490, reported efficiency ratio 1.0029 to 1.0046 | pass |
| `interval_calibration` | `v0_ate_crossfit__noise_control` | control | average treatment effect in stratum V = 0, from the cross-fitted fit: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8033 to 0.8437, SE ratio 0.6602 to 0.7125, empirical efficiency ratio 1.4089 to 1.5203, reported efficiency ratio 1.0029 to 1.0046 | pass |
| `interval_calibration` | `v0_ate_crossfit__shrunken_se_control` | control | average treatment effect in stratum V = 0, from the cross-fitted fit: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8003 to 0.8409, SE ratio 0.6692 to 0.7202, empirical efficiency ratio 0.9757 to 1.0499, reported efficiency ratio 0.7021 to 0.7032 | pass |
| `interval_calibration` | `v0_att__correctly_specified` | positive | average effect on the treated in stratum V = 0: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9369 to 0.9604, SE ratio 0.9452 to 1.0172, empirical efficiency ratio 0.9828 to 1.0582, reported efficiency ratio 0.9986 to 1.0011 | pass |
| `interval_calibration` | `v0_att__noise_control` | control | average effect on the treated in stratum V = 0: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8050 to 0.8453, SE ratio 0.6708 to 0.7213, empirical efficiency ratio 1.3863 to 1.4902, reported efficiency ratio 0.9986 to 1.0012 | pass |
| `interval_calibration` | `v0_att__shrunken_se_control` | control | average effect on the treated in stratum V = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7933 to 0.8346, SE ratio 0.6615 to 0.7119, empirical efficiency ratio 0.9831 to 1.0576, reported efficiency ratio 0.6991 to 0.7008 | pass |
| `interval_calibration` | `v0_ey0__correctly_specified` | positive | mean under no treatment in stratum V = 0: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9439 to 0.9659, SE ratio 0.9603 to 1.0366, empirical efficiency ratio 0.9647 to 1.0414, reported efficiency ratio 0.9986 to 1.0012 | pass |
| `interval_calibration` | `v0_ey0__noise_control` | control | mean under no treatment in stratum V = 0: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8238 to 0.8623, SE ratio 0.6808 to 0.7369, empirical efficiency ratio 1.3571 to 1.4689, reported efficiency ratio 0.9986 to 1.0011 | pass |
| `interval_calibration` | `v0_ey0__shrunken_se_control` | control | mean under no treatment in stratum V = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7933 to 0.8346, SE ratio 0.6728 to 0.7268, empirical efficiency ratio 0.9635 to 1.0406, reported efficiency ratio 0.6991 to 0.7008 | pass |
| `interval_calibration` | `v0_ey1__correctly_specified` | positive | mean under treatment in stratum V = 0: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9324 to 0.9567, SE ratio 0.9448 to 1.0165, empirical efficiency ratio 0.9836 to 1.0585, reported efficiency ratio 0.9984 to 1.0010 | pass |
| `interval_calibration` | `v0_ey1__noise_control` | control | mean under treatment in stratum V = 0: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7994 to 0.8402, SE ratio 0.6707 to 0.7215, empirical efficiency ratio 1.3851 to 1.4909, reported efficiency ratio 0.9984 to 1.0009 | pass |
| `interval_calibration` | `v0_ey1__shrunken_se_control` | control | mean under treatment in stratum V = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8050 to 0.8453, SE ratio 0.6619 to 0.7124, empirical efficiency ratio 0.9823 to 1.0570, reported efficiency ratio 0.6989 to 0.7007 | pass |
| `interval_calibration` | `v0_par__correctly_specified` | positive | population attributable risk in stratum V = 0: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9346 to 0.9585, SE ratio 0.9444 to 1.0161, empirical efficiency ratio 0.9853 to 1.0603, reported efficiency ratio 0.9995 to 1.0033 | pass |
| `interval_calibration` | `v0_par__noise_control` | control | population attributable risk in stratum V = 0: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8294 to 0.8675, SE ratio 0.6847 to 0.7395, empirical efficiency ratio 1.3538 to 1.4622, reported efficiency ratio 0.9995 to 1.0033 | pass |
| `interval_calibration` | `v0_par__shrunken_se_control` | control | population attributable risk in stratum V = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7894 to 0.8310, SE ratio 0.6607 to 0.7108, empirical efficiency ratio 0.9860 to 1.0606, reported efficiency ratio 0.6997 to 0.7023 | pass |
| `interval_calibration` | `v1_atc__correctly_specified` | positive | average effect on the untreated in stratum V = 1: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9374 to 0.9608, SE ratio 0.9626 to 1.0407, empirical efficiency ratio 0.9602 to 1.0379, reported efficiency ratio 0.9965 to 1.0018 | pass |
| `interval_calibration` | `v1_atc__noise_control` | control | average effect on the untreated in stratum V = 1: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8137 to 0.8532, SE ratio 0.6912 to 0.7447, empirical efficiency ratio 1.3419 to 1.4453, reported efficiency ratio 0.9966 to 1.0018 | pass |
| `interval_calibration` | `v1_atc__shrunken_se_control` | control | average effect on the untreated in stratum V = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8133 to 0.8529, SE ratio 0.6733 to 0.7274, empirical efficiency ratio 0.9614 to 1.0388, reported efficiency ratio 0.6976 to 0.7013 | pass |
| `interval_calibration` | `v1_ate__correctly_specified` | positive | average treatment effect in stratum V = 1: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9328 to 0.9570, SE ratio 0.9598 to 1.0366, empirical efficiency ratio 0.9633 to 1.0409, reported efficiency ratio 0.9970 to 1.0006 | pass |
| `interval_calibration` | `v1_ate__noise_control` | control | average treatment effect in stratum V = 1: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8190 to 0.8580, SE ratio 0.6882 to 0.7396, empirical efficiency ratio 1.3505 to 1.4514, reported efficiency ratio 0.9970 to 1.0007 | pass |
| `interval_calibration` | `v1_ate__shrunken_se_control` | control | average treatment effect in stratum V = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8142 to 0.8536, SE ratio 0.6724 to 0.7242, empirical efficiency ratio 0.9652 to 1.0401, reported efficiency ratio 0.6979 to 0.7005 | pass |
| `interval_calibration` | `v1_ate_crossfit__correctly_specified` | positive | average treatment effect in stratum V = 1, from the cross-fitted fit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9346 to 0.9585, SE ratio 0.9702 to 1.0450, empirical efficiency ratio 0.9577 to 1.0317, reported efficiency ratio 0.9991 to 1.0028 | pass |
| `interval_calibration` | `v1_ate_crossfit__noise_control` | control | average treatment effect in stratum V = 1, from the cross-fitted fit: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8042 to 0.8445, SE ratio 0.6703 to 0.7232, empirical efficiency ratio 1.3844 to 1.4932, reported efficiency ratio 0.9991 to 1.0028 | pass |
| `interval_calibration` | `v1_ate_crossfit__shrunken_se_control` | control | average treatment effect in stratum V = 1, from the cross-fitted fit: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8072 to 0.8473, SE ratio 0.6796 to 0.7323, empirical efficiency ratio 0.9562 to 1.0312, reported efficiency ratio 0.6994 to 0.7019 | pass |
| `interval_calibration` | `v1_att__correctly_specified` | positive | average effect on the treated in stratum V = 1: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9369 to 0.9604, SE ratio 0.9637 to 1.0389, empirical efficiency ratio 0.9653 to 1.0406, reported efficiency ratio 0.9999 to 1.0055 | pass |
| `interval_calibration` | `v1_att__noise_control` | control | average effect on the treated in stratum V = 1: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8116 to 0.8513, SE ratio 0.6806 to 0.7340, empirical efficiency ratio 1.3663 to 1.4739, reported efficiency ratio 0.9998 to 1.0056 | pass |
| `interval_calibration` | `v1_att__shrunken_se_control` | control | average effect on the treated in stratum V = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8137 to 0.8532, SE ratio 0.6733 to 0.7285, empirical efficiency ratio 0.9640 to 1.0428, reported efficiency ratio 0.6999 to 0.7039 | pass |
| `interval_calibration` | `v1_ey0__correctly_specified` | positive | mean under no treatment in stratum V = 1: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9337 to 0.9578, SE ratio 0.9594 to 1.0352, empirical efficiency ratio 0.9659 to 1.0424, reported efficiency ratio 0.9981 to 1.0018 | pass |
| `interval_calibration` | `v1_ey0__noise_control` | control | mean under no treatment in stratum V = 1: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8225 to 0.8612, SE ratio 0.6853 to 0.7392, empirical efficiency ratio 1.3522 to 1.4590, reported efficiency ratio 0.9981 to 1.0018 | pass |
| `interval_calibration` | `v1_ey0__shrunken_se_control` | control | mean under no treatment in stratum V = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8046 to 0.8449, SE ratio 0.6713 to 0.7250, empirical efficiency ratio 0.9660 to 1.0422, reported efficiency ratio 0.6987 to 0.7012 | pass |
| `interval_calibration` | `v1_ey1__correctly_specified` | positive | mean under treatment in stratum V = 1: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9241 to 0.9499, SE ratio 0.9588 to 1.0342, empirical efficiency ratio 0.9634 to 1.0385, reported efficiency ratio 0.9923 to 0.9995 | pass |
| `interval_calibration` | `v1_ey1__noise_control` | control | mean under treatment in stratum V = 1: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8046 to 0.8449, SE ratio 0.6733 to 0.7247, empirical efficiency ratio 1.3737 to 1.4788, reported efficiency ratio 0.9923 to 0.9994 | pass |
| `interval_calibration` | `v1_ey1__shrunken_se_control` | control | mean under treatment in stratum V = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7994 to 0.8402, SE ratio 0.6709 to 0.7238, empirical efficiency ratio 0.9638 to 1.0384, reported efficiency ratio 0.6946 to 0.6996 | pass |
| `interval_calibration` | `v1_par__correctly_specified` | positive | population attributable risk in stratum V = 1: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9374 to 0.9608, SE ratio 0.9601 to 1.0378, empirical efficiency ratio 0.9643 to 1.0422, reported efficiency ratio 0.9980 to 1.0028 | pass |
| `interval_calibration` | `v1_par__noise_control` | control | population attributable risk in stratum V = 1: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8085 to 0.8485, SE ratio 0.6765 to 0.7285, empirical efficiency ratio 1.3732 to 1.4788, reported efficiency ratio 0.9980 to 1.0029 | pass |
| `interval_calibration` | `v1_par__shrunken_se_control` | control | population attributable risk in stratum V = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8116 to 0.8513, SE ratio 0.6721 to 0.7275, empirical efficiency ratio 0.9627 to 1.0420, reported efficiency ratio 0.6986 to 0.7020 | pass |
| `interval_calibration` | `v2_atc__correctly_specified` | positive | average effect on the untreated in stratum V = 2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9255 to 0.9511, SE ratio 0.9344 to 1.0090, empirical efficiency ratio 0.9874 to 1.0663, reported efficiency ratio 0.9908 to 1.0016 | pass |
| `interval_calibration` | `v2_atc__noise_control` | control | average effect on the untreated in stratum V = 2: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7994 to 0.8402, SE ratio 0.6611 to 0.7123, empirical efficiency ratio 1.3986 to 1.5076, reported efficiency ratio 0.9912 to 1.0018 | pass |
| `interval_calibration` | `v2_atc__shrunken_se_control` | control | average effect on the untreated in stratum V = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7985 to 0.8394, SE ratio 0.6543 to 0.7063, empirical efficiency ratio 0.9876 to 1.0657, reported efficiency ratio 0.6938 to 0.7010 | pass |
| `interval_calibration` | `v2_ate__correctly_specified` | positive | average treatment effect in stratum V = 2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9250 to 0.9507, SE ratio 0.9355 to 1.0108, empirical efficiency ratio 0.9847 to 1.0647, reported efficiency ratio 0.9910 to 1.0000 | pass |
| `interval_calibration` | `v2_ate__noise_control` | control | average treatment effect in stratum V = 2: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8007 to 0.8413, SE ratio 0.6654 to 0.7169, empirical efficiency ratio 1.3881 to 1.4954, reported efficiency ratio 0.9911 to 1.0000 | pass |
| `interval_calibration` | `v2_ate__shrunken_se_control` | control | average treatment effect in stratum V = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8024 to 0.8429, SE ratio 0.6546 to 0.7074, empirical efficiency ratio 0.9846 to 1.0653, reported efficiency ratio 0.6937 to 0.7000 | pass |
| `interval_calibration` | `v2_ate_crossfit__correctly_specified` | positive | average treatment effect in stratum V = 2, from the cross-fitted fit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9273 to 0.9526, SE ratio 0.9324 to 1.0031, empirical efficiency ratio 0.9998 to 1.0762, reported efficiency ratio 0.9986 to 1.0078 | pass |
| `interval_calibration` | `v2_ate_crossfit__noise_control` | control | average treatment effect in stratum V = 2, from the cross-fitted fit: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8129 to 0.8525, SE ratio 0.6783 to 0.7290, empirical efficiency ratio 1.3763 to 1.4784, reported efficiency ratio 0.9985 to 1.0078 | pass |
| `interval_calibration` | `v2_ate_crossfit__shrunken_se_control` | control | average treatment effect in stratum V = 2, from the cross-fitted fit: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7855 to 0.8274, SE ratio 0.6523 to 0.7009, empirical efficiency ratio 1.0013 to 1.0771, reported efficiency ratio 0.6989 to 0.7054 | pass |
| `interval_calibration` | `v2_att__correctly_specified` | positive | average effect on the treated in stratum V = 2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9324 to 0.9567, SE ratio 0.9478 to 1.0249, empirical efficiency ratio 0.9722 to 1.0515, reported efficiency ratio 0.9922 to 1.0015 | pass |
| `interval_calibration` | `v2_att__noise_control` | control | average effect on the treated in stratum V = 2: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8059 to 0.8461, SE ratio 0.6781 to 0.7297, empirical efficiency ratio 1.3667 to 1.4695, reported efficiency ratio 0.9923 to 1.0016 | pass |
| `interval_calibration` | `v2_att__shrunken_se_control` | control | average effect on the treated in stratum V = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8055 to 0.8457, SE ratio 0.6633 to 0.7171, empirical efficiency ratio 0.9733 to 1.0518, reported efficiency ratio 0.6946 to 0.7010 | pass |
| `interval_calibration` | `v2_ey0__correctly_specified` | positive | mean under no treatment in stratum V = 2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9356 to 0.9593, SE ratio 0.9507 to 1.0255, empirical efficiency ratio 0.9711 to 1.0488, reported efficiency ratio 0.9923 to 0.9999 | pass |
| `interval_calibration` | `v2_ey0__noise_control` | control | mean under no treatment in stratum V = 2: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7955 to 0.8366, SE ratio 0.6742 to 0.7241, empirical efficiency ratio 1.3759 to 1.4771, reported efficiency ratio 0.9926 to 1.0000 | pass |
| `interval_calibration` | `v2_ey0__shrunken_se_control` | control | mean under no treatment in stratum V = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8103 to 0.8501, SE ratio 0.6639 to 0.7182, empirical efficiency ratio 0.9712 to 1.0493, reported efficiency ratio 0.6947 to 0.6999 | pass |
| `interval_calibration` | `v2_ey1__correctly_specified` | positive | mean under treatment in stratum V = 2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9056 to 0.9344, SE ratio 0.9315 to 1.0094, empirical efficiency ratio 0.9753 to 1.0552, reported efficiency ratio 0.9742 to 0.9931 | **fail** |
| `interval_calibration` | `v2_ey1__noise_control` | control | mean under treatment in stratum V = 2: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7972 to 0.8382, SE ratio 0.6667 to 0.7212, empirical efficiency ratio 1.3655 to 1.4744, reported efficiency ratio 0.9744 to 0.9932 | pass |
| `interval_calibration` | `v2_ey1__shrunken_se_control` | control | mean under treatment in stratum V = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7955 to 0.8366, SE ratio 0.6517 to 0.7058, empirical efficiency ratio 0.9769 to 1.0558, reported efficiency ratio 0.6821 to 0.6954 | pass |
| `interval_calibration` | `v2_par__correctly_specified` | positive | population attributable risk in stratum V = 2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9356 to 0.9593, SE ratio 0.9525 to 1.0279, empirical efficiency ratio 0.9704 to 1.0476, reported efficiency ratio 0.9926 to 1.0026 | pass |
| `interval_calibration` | `v2_par__noise_control` | control | population attributable risk in stratum V = 2: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8007 to 0.8413, SE ratio 0.6634 to 0.7141, empirical efficiency ratio 1.3974 to 1.5047, reported efficiency ratio 0.9926 to 1.0027 | pass |
| `interval_calibration` | `v2_par__shrunken_se_control` | control | population attributable risk in stratum V = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8085 to 0.8485, SE ratio 0.6668 to 0.7195, empirical efficiency ratio 0.9695 to 1.0471, reported efficiency ratio 0.6949 to 0.7018 | pass |
| `simultaneous_coverage` | `crossfit_strata__pointwise_joint_control` | control | the three marginal and nine stratum parameters of the cross-fitted fit: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.5999 to 0.6512 | pass |
| `simultaneous_coverage` | `crossfit_strata__simultaneous_band` | positive | the three marginal and nine stratum parameters of the cross-fitted fit: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9038 to 0.9329 | **fail** |
| `simultaneous_coverage` | `strata__pointwise_joint_control` | control | the six marginal and eighteen stratum parameters of the in-sample fit: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.5688 to 0.6208 | pass |
| `simultaneous_coverage` | `strata__simultaneous_band` | positive | the six marginal and eighteen stratum parameters of the in-sample fit: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.8881 to 0.9194 | **fail** |
| `stratum_targeting_necessity` | `v0_ate__marginal_fluctuation` | control | average treatment effect in stratum V = 0: the same outcome regression with one marginal fluctuation per arm, averaged inside each stratum | bias interval must fall entirely outside the margin | bias -0.0371 to -0.0341, margin 0.0052 | pass |
| `stratum_targeting_necessity` | `v0_ate__stratified` | positive | average treatment effect in stratum V = 0: the stratified fit with an outcome regression that omits the stratum, so the per-stratum score blocks do all the adjusting | bias interval inside the equivalence margin | bias -0.0023 to 0.0023, margin 0.0077 | pass |
| `stratum_targeting_necessity` | `v2_ate__marginal_fluctuation` | control | average treatment effect in stratum V = 2: the same outcome regression with one marginal fluctuation per arm, averaged inside each stratum | bias interval must fall entirely outside the margin | bias 0.0758 to 0.0784, margin 0.0044 | pass |
| `stratum_targeting_necessity` | `v2_ate__stratified` | positive | average treatment effect in stratum V = 2: the stratified fit with an outcome regression that omits the stratum, so the per-stratum score blocks do all the adjusting | bias interval inside the equivalence margin | bias -0.0021 to 0.0026, margin 0.0079 | pass |
<!-- /generated -->

## Result

The study publishes under the `reporting` policy. The declared run had five red verdicts, and the
[red-cell ledger](red-cells.md) names `strata-boundary-mean` as their owner. Every other primary
test, paired comparison and property cell passed.

| verdict | result | reading |
| --- | --- | --- |
| paired comparison with `tmle3` | 9 of 9 equivalent | the stratum arm means and ATEs agree with `tmle_stratified` |
| primary truth test of `ey[1][V=2]`, both implementations | red. Coverage 0.9170 for each, with the lower 99% endpoint below the 0.90 floor | the bias is inside its margin and the SE ratio is 0.985. The treated mean of stratum 2 is 0.9358. At n = 2,000 the stratum holds about 157 treated rows and about 9 expected non-events, so its Wald interval under-covers, and `tmle3` shows the same coverage on the same draws |
| `v2_ey1` calibration cell | red. Coverage 0.9208 | the same mean, in the property sample |
| `strata` and `crossfit_strata` bands | red. Joint coverage 0.9046 and 0.9192 | the declared diagnostic read the same rows at the design critical values, 2.846 and 2.778. That oracle band covers 0.9121 and 0.9258, so it also under-covers, and the multiplier is not the main cause. The package critical values average 2.816 and 2.750, which account for the 0.0075 and 0.0066 between the two coverages. Each band inherits the shortfall of the pointwise interval above |
| the other 17 calibration cells, three cross-fitted cells, and their controls | pass | every stratum parameter, ATT, ATC and PAR included, is calibrated at its efficiency bound |
| double robustness and stratum targeting | pass | the marginal-fluctuation control moves at least 1.1515 empirical SDs in each outer stratum, and the stratified fit stays inside its margin |

The route is the finite-sample one. The default band stays on, because nothing in the band's
construction failed. The study module records the diagnostic, and
[`tests/unit/test_band_shortfall_reading.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_band_shortfall_reading.py) rebuilds
every number of it from the committed rows. [`strata-boundary-mean`](../../roadmap.md#red-cell-owners) owns the five red verdicts.

## Refusals

| refusal | pinning test |
| --- | --- |
| `cv_evaluation=True` or `targeting_scheme="fold"` | `tests/unit/test_natural_course_crossfit.py::test_a_complete_outcome_fit_with_strata_meets_the_generic_strata_gate` |
| an incremental stratum without a treatment arm, an MSM singular inside a stratum, or a `DRTMLE` stratum with no trainable rows | `tests/unit/test_stratified_incremental_exact.py`, `tests/unit/test_stratified_msm_exact.py`, `tests/unit/test_refusals_before_the_nuisance_fit.py` |
| the cross-fitted arm-indexed missing-outcome contract, and the cross-fitted missing-outcome natural-course mean | `tests/unit/test_stratified_refusals.py` |
| a stratum parameter in an omitted-variable or missingness sensitivity analysis | `tests/unit/test_stratified_targets.py` |

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 18 | primary tests against truth |
| `independent_tests_passed` | 16 | of those, passing |
| `paired_tests_total` | 9 | external comparisons declared |
| `paired_tests_passed` | 9 | external comparisons passing |
| `property_cells_total` | 83 | repeated-sampling property cells |
| `property_cells_passed` | 80 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0626 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9170 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9160 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0844 | highest bootstrap primary SE-ratio endpoint |
| `properties[simultaneous_coverage/strata__simultaneous_band]:coverage` | 0.9046 | joint coverage of the in-sample band over 24 parameters |
| `properties[simultaneous_coverage/strata__simultaneous_band]:coverage_ci_upper` | 0.9194 | its 99% upper endpoint |
| `properties[simultaneous_coverage/strata__pointwise_joint_control]:coverage` | 0.5950 | joint coverage of the pointwise intervals of the same fits |
| `properties[simultaneous_coverage/crossfit_strata__simultaneous_band]:coverage` | 0.9192 | joint coverage of the cross-fitted band over 12 parameters |
| `properties[simultaneous_coverage/crossfit_strata__simultaneous_band]:coverage_ci_upper` | 0.9329 | its 99% upper endpoint |
| `properties[simultaneous_coverage/crossfit_strata__pointwise_joint_control]:coverage` | 0.6258 | joint coverage of the cross-fitted pointwise intervals |
| `properties[interval_calibration/v2_ey1__correctly_specified]:coverage` | 0.9208 | coverage of the treated mean in the smallest stratum |
| `properties[stratum_targeting_necessity/v0_ate__stratified]:targeting_displacement` | 1.1515 | displacement of the marginal-fluctuation control, the less displaced outer stratum |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration and joint-coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration and joint-coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest supported type-I rate. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | factor the shrunken-SE control multiplies the standard error by |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio band, lower limit |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio band, upper limit |
| `margin:targeting_displacement` | 0.2500 | smallest displacement of the marginal-fluctuation control, in empirical SDs |
| `bound:ate[V=0]_standard_error` | 0.0308 | exact efficiency bound on the standard error of `ate[V=0]` at this n |
| `bound:ate[V=1]_standard_error` | 0.0347 | exact efficiency bound on the standard error of `ate[V=1]` at this n |
| `bound:ate[V=2]_standard_error` | 0.0309 | exact efficiency bound on the standard error of `ate[V=2]` at this n |
| `bound:ey[0][V=0]_standard_error` | 0.0226 | exact efficiency bound on the standard error of `ey[0][V=0]` at this n |
| `bound:ey[0][V=1]_standard_error` | 0.0259 | exact efficiency bound on the standard error of `ey[0][V=1]` at this n |
| `bound:ey[0][V=2]_standard_error` | 0.0228 | exact efficiency bound on the standard error of `ey[0][V=2]` at this n |
| `bound:ey[1][V=0]_standard_error` | 0.0216 | exact efficiency bound on the standard error of `ey[1][V=0]` at this n |
| `bound:ey[1][V=1]_standard_error` | 0.0235 | exact efficiency bound on the standard error of `ey[1][V=1]` at this n |
| `bound:ey[1][V=2]_standard_error` | 0.0212 | exact efficiency bound on the standard error of `ey[1][V=2]` at this n |

## Limitations

| limitation | what it means for use |
| --- | --- |
| One law with three strata and a binary treatment and outcome | A continuous outcome, more strata, or a multi-arm treatment is not measured |
| Main-terms logistic learners only | A flexible learner is not covered |
| No weights, clusters, missing outcomes or repeats | Each composition with strata is a separate study |
| ATT, ATC and PAR have no comparator | They are measured against the exact truth only |
| The stratum-targeting control reads the outer strata | The middle stratum's control could not fail. The exact-law test checks that stratum's score block |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_stratified/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_stratified/manifest.json)
records the seeds, margins, laws, source hashes, and result hashes.
