# A declared treatment mechanism beside sample-reading weights

This study validates `mechanism_law="sample"`: point `TMLE` and `LTMLE` on weighted data that
declares the treatment mechanism of the rows as drawn with `treatment_probabilities=`. The fit
estimates the mechanism of the weight-tilted law as
$\hat g_w(a\mid W) \propto g_0(a\mid W)\,\hat r(a\mid W)$, where $\hat r$ is the treatment
learner's classification of $A$ on $W$ with weights $w/g_0(A\mid W)$. It is an
estimated-mechanism fit, so its interval needs the outcome regression and the tilt classifier
both consistent at a product rate. The section on the
[known treatment mechanism](../point-treatment-tmle.md#known-treatment-mechanism) states the
construction and its remainder, and the section on
[known node mechanisms](../longitudinal-tmle.md#known-node-mechanisms) states it node by node.

The declared run passed every gated verdict, with no failed fit: 3 of 3 primary rows and 26 of
26 property cells. No external package implements the tilted mechanism from a declaration, so
no canonical implementation is compared. The fast tier's exact-law tests
(`tests/unit/test_known_mechanism_weights.py` and
`tests/unit/test_known_mechanism_weights_longitudinal.py`) pin the construction.

## The laws

**SW** is a point law whose weight reads $(A, W)$ only, so the tilted ATE keeps its causal
reading. $W_1 \sim N(0, 1)$ and $W_2 \sim \operatorname{Bernoulli}(0.5)$. The declared
randomization is $g_0(1\mid W) = 0.3 + 0.4 W_2$. The outcome is
$Y \sim \operatorname{Bernoulli}(\operatorname{expit}(-1 + A + W_1 + 1.5 A W_1 - W_1^2/2 +
0.5 W_2))$ and the weight is $w = \exp(0.4 A + 0.5 A W_1 + 0.15 W_1)$. Then
$E[w \mid A, W] = w$ and the tilt is $r(1\mid W) = \operatorname{expit}(0.4 + 0.5 W_1)$, so a
main-terms logistic classifier is correct for it. The truth is $\Psi(P_w) = 0.330353$ against an
unweighted ATE of 0.234120, and the Kish effective sample size is 0.65 n.

**LW** is a two-node law. $L_0 \sim N(0, 1)$, $A_1 \sim
\operatorname{Bernoulli}(\operatorname{expit}(0.4 L_0))$ (declared), $L_1 \sim N(0.5 L_0 +
0.5 A_1, 1)$, $A_2 \sim \operatorname{Bernoulli}(\operatorname{expit}(-0.2 + 0.6 L_1))$
(declared), and $Y = 0.5 L_0 + 0.4 A_1 + 0.5 A_1 L_0 + 0.6 L_1 + 0.5 A_2 + 0.3 A_1 A_2 +
N(0, 1)$. The weight $w = \exp(0.4 A_1 + 0.5 A_1 L_0 + 0.15 L_0)$ tilts the first node and not
the second. The target is the contrast of always and never treating, 1.769979 under $P_w$
against 1.5 in the rows as drawn, and the Kish effective sample size is 0.63 n.

`tests/studies/known_sample_weights_law.py` defines both laws.
`tests/unit/test_known_sample_weights_design.py` recomputes every truth by quadrature or a
longhand g-formula and checks it against Monte Carlo, and recomputes each control's exact-law
bias from the population targeting step.

## The fits

| fit | law | reading | outcome regression | tilt classifier |
| --- | --- | --- | --- | --- |
| `sample_correct` | SW | `"sample"` | logistic on $(A, W_1, A W_1, W_1^2, W_2)$, correct | main-terms logistic on $(W_1, W_2)$, correct |
| `sample_main` | SW | `"sample"` | main-terms logistic on $(A, W_1, W_2)$, wrong | correct |
| `tilt_wrong` | SW | `"sample"` | correct | logistic on $W_2$ alone, wrong |
| `both_wrong` | SW | `"sample"` | wrong | wrong |
| `target_main` | SW | `"target"`, dividing by $g_0$ | wrong | none |
| `ltmle_sample` | LW | `"sample"` | weighted least squares at each node, correct | main-terms logistic per node, correct |
| `ltmle_sample_flat` | LW | `"sample"` | intercept-only at node one, wrong | correct |
| `ltmle_target_flat` | LW | `"target"` | intercept-only at node one, wrong | none |

Every point fit cross-fits on five folds with `g_bounds=(0.01, 0.99)`, which no tilted mechanism
reaches. Every LTMLE fit cross-fits on five folds with `q_bounds=(-25, 25)`, which no draw
reaches, and `g_bounds=(1e-4, 1.0)`, which no cumulative product reaches.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a binary outcome, W1 normal and W2 binary, a declared randomization by W2 and a weight that reads the arm and W1; mechanism_law='sample' with a correct logistic outcome regression and a correct main-terms tilt classifier, five folds | `ate` | average treatment effect | `cleverly` | -0.0013 to 0.000678 | 0.9545 | 1.0021 | pass |
| a binary outcome, W1 normal and W2 binary, a declared randomization by W2 and a weight that reads the arm and W1; mechanism_law='sample' with a correct logistic outcome regression and a correct main-terms tilt classifier, five folds | `ey0` | counterfactual mean under no treatment | `cleverly` | -0.000821 to 0.000790 | 0.9575 | 1.0074 | pass |
| a binary outcome, W1 normal and W2 binary, a declared randomization by W2 and a weight that reads the arm and W1; mechanism_law='sample' with a correct logistic outcome regression and a correct main-terms tilt classifier, five folds | `ey1` | counterfactual mean under treatment | `cleverly` | -0.000923 to 0.000288 | 0.9405 | 0.9770 | pass |
<!-- /generated -->

The primary rows draw 2,000 samples of 4,000 rows from SW and fit `sample_correct`.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `both_misspecified` | `sw_n4000` | control | law SW at n = 4,000, the ATE, mechanism_law='sample' with the main-terms outcome regression on (A, W1, W2) and a tilt classifier on W2 alone | 99% interval of the standardized bias wholly outside the margin | bias -0.0354 to -0.0340, margin 0.0042 | pass |
| `interval_calibration` | `lw_n4000__correctly_specified` | positive | law LW at n = 4,000, the always-versus-never contrast, mechanism_law='sample': both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9426 to 0.9603, SE ratio 0.9575 to 1.0263 | pass |
| `interval_calibration` | `lw_n4000__noise_control` | control | law LW at n = 4,000, the always-versus-never contrast, mechanism_law='sample': a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8248 to 0.8549, SE ratio 0.6957 to 0.7384 | pass |
| `interval_calibration` | `lw_n4000__shrunken_se_control` | control | law LW at n = 4,000, the always-versus-never contrast, mechanism_law='sample': the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8170 to 0.8477, SE ratio 0.6707 to 0.7180 | pass |
| `interval_calibration` | `sw_n1000__correctly_specified` | positive | law SW at n = 1,000, the ATE, mechanism_law='sample': both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9361 to 0.9548, SE ratio 0.9653 to 1.0236 | pass |
| `interval_calibration` | `sw_n1000__noise_control` | control | law SW at n = 1,000, the ATE, mechanism_law='sample': a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8163 to 0.8469, SE ratio 0.6912 to 0.7303 | pass |
| `interval_calibration` | `sw_n1000__shrunken_se_control` | control | law SW at n = 1,000, the ATE, mechanism_law='sample': the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8049 to 0.8363, SE ratio 0.6756 to 0.7167 | pass |
| `interval_calibration` | `sw_n16000__correctly_specified` | positive | law SW at n = 16,000, the ATE, mechanism_law='sample': both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9410 to 0.9589, SE ratio 0.9720 to 1.0295 | pass |
| `interval_calibration` | `sw_n16000__noise_control` | control | law SW at n = 16,000, the ATE, mechanism_law='sample': a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8232 to 0.8534, SE ratio 0.6873 to 0.7281 | pass |
| `interval_calibration` | `sw_n16000__shrunken_se_control` | control | law SW at n = 16,000, the ATE, mechanism_law='sample': the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8150 to 0.8457, SE ratio 0.6797 to 0.7213 | pass |
| `interval_calibration` | `sw_n4000__correctly_specified` | positive | law SW at n = 4,000, the ATE, mechanism_law='sample': both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9418 to 0.9596, SE ratio 0.9734 to 1.0307 | pass |
| `interval_calibration` | `sw_n4000__noise_control` | control | law SW at n = 4,000, the ATE, mechanism_law='sample': a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8160 to 0.8467, SE ratio 0.6938 to 0.7341 | pass |
| `interval_calibration` | `sw_n4000__shrunken_se_control` | control | law SW at n = 4,000, the ATE, mechanism_law='sample': the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8124 to 0.8433, SE ratio 0.6812 to 0.7204 | pass |
| `sample_bias` | `lw_n4000` | positive | law LW at n = 4,000, the always-versus-never contrast, mechanism_law='sample' with every nuisance correct | 99% interval of the standardized bias inside the margin | bias -0.0025 to 0.0030, margin 0.0167 | pass |
| `sample_bias` | `sw_n1000` | positive | law SW at n = 1,000, the ATE, mechanism_law='sample' with every nuisance correct | 99% interval of the standardized bias inside the margin | bias -0.000736 to 0.0021, margin 0.0087 | pass |
| `sample_bias` | `sw_n16000` | positive | law SW at n = 16,000, the ATE, mechanism_law='sample' with every nuisance correct | 99% interval of the standardized bias inside the margin | bias -0.000283 to 0.000408, margin 0.0021 | pass |
| `sample_bias` | `sw_n4000` | positive | law SW at n = 4,000, the ATE, mechanism_law='sample' with every nuisance correct | 99% interval of the standardized bias inside the margin | bias -0.000656 to 0.000733, margin 0.0043 | pass |
| `sample_misspecified` | `lw_n4000` | positive | law LW at n = 4,000, the always-versus-never contrast, mechanism_law='sample' with an intercept-only regression at node one and the correct tilt classifier | 99% interval of the standardized bias inside the margin | bias -0.0050 to 0.000690, margin 0.0173 | pass |
| `sample_misspecified` | `sw_n1000` | positive | law SW at n = 1,000, the ATE, mechanism_law='sample' with the main-terms outcome regression on (A, W1, W2) and the correct tilt classifier | 99% interval of the standardized bias inside the margin | bias -0.000320 to 0.0027, margin 0.0092 | pass |
| `sample_misspecified` | `sw_n16000` | positive | law SW at n = 16,000, the ATE, mechanism_law='sample' with the main-terms outcome regression on (A, W1, W2) and the correct tilt classifier | 99% interval of the standardized bias inside the margin | bias -0.000300 to 0.000447, margin 0.0023 | pass |
| `sample_misspecified` | `sw_n4000` | positive | law SW at n = 4,000, the ATE, mechanism_law='sample' with the main-terms outcome regression on (A, W1, W2) and the correct tilt classifier | 99% interval of the standardized bias inside the margin | bias -0.000507 to 0.000974, margin 0.0045 | pass |
| `target_reading_control` | `lw_n4000` | control | law LW at n = 4,000, the always-versus-never contrast, mechanism_law='target' (dividing by the declaration) with an intercept-only regression at node one | 99% interval of the standardized bias wholly outside the margin | bias 0.4524 to 0.4579, margin 0.0167 | pass |
| `target_reading_control` | `sw_n1000` | control | law SW at n = 1,000, the ATE, mechanism_law='target' (dividing by the declaration) with the main-terms outcome regression on (A, W1, W2) | 99% interval of the standardized bias wholly outside the margin | bias -0.0336 to -0.0309, margin 0.0082 | pass |
| `target_reading_control` | `sw_n16000` | control | law SW at n = 16,000, the ATE, mechanism_law='target' (dividing by the declaration) with the main-terms outcome regression on (A, W1, W2) | 99% interval of the standardized bias wholly outside the margin | bias -0.0333 to -0.0326, margin 0.0021 | pass |
| `target_reading_control` | `sw_n4000` | control | law SW at n = 4,000, the ATE, mechanism_law='target' (dividing by the declaration) with the main-terms outcome regression on (A, W1, W2) | 99% interval of the standardized bias wholly outside the margin | bias -0.0336 to -0.0323, margin 0.0041 | pass |
| `tilt_misspecified` | `sw_n4000` | positive | law SW at n = 4,000, the ATE, mechanism_law='sample' with the correct outcome regression and a tilt classifier on W2 alone | 99% interval of the standardized bias inside the margin | bias -0.000819 to 0.000540, margin 0.0042 | pass |
<!-- /generated -->

`sample_misspecified` is the direct witness of the tilt. With the outcome regression wrong,
only a correct tilted mechanism removes the bias, so a reverted tilt or a partial one shows
there as bias. Its interval over-covers on LW, because the curve omits the term for estimating
$\hat g_w$, so its calibration is published and not gated. `tilt_misspecified` witnesses the
other half of the double robustness, and `both_misspecified` shows that its wrong classifier
drops a term that matters.

Each control row also publishes its exact-law bias and its paired difference: the control minus
the `sample_misspecified` cell of the same draws, in units of the control's standard deviation.
The partner removes the whole control bias on the same draws.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2000 | primary replications |
| `n` | 4000 | observations per primary replication |
| `independent_tests_passed` | 3 | truth tests passing |
| `independent_tests_total` | 3 | truth tests reported |
| `property_cells_passed` | 26 | property cells passing |
| `property_cells_total` | 26 | property cells reported |
| `max_standardized_bias` | 0.0303 | largest primary standardized bias |
| `min_coverage` | 0.9405 | lowest primary coverage |
| `properties[interval_calibration/sw_n1000__correctly_specified]:se_ratio` | 0.9934 | SE ratio of the SW calibration cell at n = 1,000 |
| `properties[interval_calibration/sw_n16000__correctly_specified]:se_ratio` | 1.0001 | SE ratio of the SW calibration cell at n = 16,000 |
| `properties[interval_calibration/lw_n4000__correctly_specified]:se_ratio` | 0.9926 | SE ratio of the LW calibration cell |
| `properties[interval_calibration/lw_n4000__correctly_specified]:coverage` | 0.9520 | coverage of the LW calibration cell |
| `properties[sample_misspecified/lw_n4000]:se_ratio` | 1.2144 | SE ratio of the LW cell with the node-one regression wrong |
| `properties[tilt_misspecified/sw_n4000]:se_ratio` | 0.9312 | SE ratio of the SW cell with the tilt classifier wrong |
| `properties[target_reading_control/sw_n16000]:standardized_bias` | -3.9599 | standardized bias of the `"target"` control on SW at n = 16,000 |
| `properties[target_reading_control/lw_n4000]:standardized_bias` | 6.8119 | standardized bias of the `"target"` control on LW |
| `properties[both_misspecified/sw_n4000]:standardized_bias` | -2.0607 | standardized bias of the both-wrong control |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:subject_failures_allowed` | 20 | the same bound, in primary replications |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |

## Limitations

| limit | what it means for use |
| --- | --- |
| parametric learners | the evidence covers correct or misspecified logistic and linear learners. With a flexible learner the ordinary conditions of an estimated-mechanism TMLE apply, and no cell measures them |
| a weight on the arm and the covariates | no cell reads a weight that depends on the outcome, which also loses the causal reading under `"sample"` |
| one node tilted on LW | the LW weight tilts the first node only. A tilt at the second node shows here only through the SE ratio, and the fast tier's exact-law tests cover it |
| binary treatments with complete data | more than two arms, a missing outcome, censoring and clusters rest on the exact-law tests alone |
| not efficient for a known $g_0$ | the fit uses the declaration as an offset of the classifier and is not efficient in the model in which $g_0$ is known |
| no comparator | no external implementation estimates the tilted mechanism from a declaration |

## Reproduction

Run `python tests/canonical/known_sample_weights/regenerate.py --output DIR` with an empty
scratch directory outside the repository to regenerate the artifacts. The pilot is
`tests/studies/known_sample_weights_pilot.py`, and the red-cell readings are declared in
`tests/canonical/known_sample_weights/regenerate.py` before any run.
