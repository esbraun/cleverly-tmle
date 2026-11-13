# Per-arm outcome regressions on point TMLE

This study validates point `TMLE` fitted with `outcome_by_arm=True`, which fits one outcome
regression per treatment arm on that arm's rows and the covariates alone. The estimand and the
targeting step do not change: the per-arm predictions form $\bar Q(a, W)$, and the fit is
targeted and its interval computed as for one pooled regression. The
[per-arm outcome regression section](../point-treatment-tmle.md#per-arm-outcome-regression)
states the option. The laws give each arm its own outcome index, so a pooled main-terms
regression on $(A, W)$ misses the arm-by-covariate interaction while one main-terms regression
per arm is correct.

No maintained package exposes this option with a TMLE interval, so
no canonical implementation is compared. The study rests on exact truths and on repeated-sampling properties.

No cell of the study could show that the option was honoured. Under the study's correct
treatment mechanism a pooled fit is also unbiased, because targeting removes the outcome
regression's first-order error, and its interval is close to the per-arm one (cells
`pooled_outcome`). The evidence that each arm's regression reads only its own rows is the
exact-law witness in `tests/unit/test_option_outcome_by_arm_witness.py`: with an intercept-only
mechanism the per-arm fit returns the truth to $10^{-10}$, the pooled fit misses it by its
predicted limit, and mutations that ignore the option or read the other arm's model fail it. A
spy learner there also checks that each arm learner sees its arm's rows and two columns.

## What was fitted

| law | outcome given $A = a$ and $W$ | per-arm learner | fits |
| --- | --- | --- | --- |
| LB | $\operatorname{Bernoulli}(\operatorname{expit}(\eta_a))$ | logistic GLM on $(W_1, W_2)$ by Newton | in sample, and stacked over five folds |
| LB-c | $N(\eta_a, 1)$ | linear regression on $(W_1, W_2)$ | in sample |
| LB-b | $\operatorname{Beta}(24 m, 24 (1 - m))$, $m = \operatorname{expit}(\eta_a)$ | fractional logistic GLM by Newton, with `q_bounds=(0, 1)` declared | stacked over five folds |

$W_1, W_2$ are independent standard normal, $g_0 = \operatorname{expit}(0.4 W_1 + 0.4 W_2)$,
$\eta_1 = -0.2 + 0.9 W_1 + 0.2 W_2$ and $\eta_0 = -0.5 + 0.1 W_1 + 0.7 W_2$. The treatment
mechanism is a logistic GLM on $(W_1, W_2)$, correct on every law. Every cell reads the ATE at
n = 2,000, and the primary phase reads the arm means and the ATE on LB in sample. The laws,
learners and exact limits are in `tests/studies/_option_by_arm_law.py`, and
`tests/unit/test_option_outcome_by_arm_design.py` recomputes the truths, the efficiency bounds
and every exact limit.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two standard-normal covariates, a binary outcome whose regression differs by arm (law LB); one main-terms logistic GLM per arm and a logistic treatment GLM, in sample, n = 2,000 | `ate` | average treatment effect | `cleverly` | -0.0017 to 0.000489 | 0.9486 | 0.9910 | pass |
| two standard-normal covariates, a binary outcome whose regression differs by arm (law LB); one main-terms logistic GLM per arm and a logistic treatment GLM, in sample, n = 2,000 | `ey0` | counterfactual mean under no treatment | `cleverly` | -0.000539 to 0.0010 | 0.9454 | 0.9936 | pass |
| two standard-normal covariates, a binary outcome whose regression differs by arm (law LB); one main-terms logistic GLM per arm and a logistic treatment GLM, in sample, n = 2,000 | `ey1` | counterfactual mean under treatment | `cleverly` | -0.0011 to 0.000412 | 0.9457 | 0.9883 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0732 to 0.0761, margin 0.0056, SE ratio 0.9789 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0013 to 0.0015, margin 0.0054, SE ratio 1.0590 | pass |
| `interval_calibration` | `in_sample_binary__correctly_specified` | positive | law LB (binary outcome), in sample, the ATE, n = 2,000: one main-terms regression per arm, each correct: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9457 to 0.9595, SE ratio 0.9801 to 1.0263 | pass |
| `interval_calibration` | `in_sample_binary__noise_control` | control | law LB (binary outcome), in sample, the ATE, n = 2,000: one main-terms regression per arm, each correct: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8214 to 0.8456, SE ratio 0.6884 to 0.7199 | pass |
| `interval_calibration` | `in_sample_binary__shrunken_se_control` | control | law LB (binary outcome), in sample, the ATE, n = 2,000: one main-terms regression per arm, each correct: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8129 to 0.8375, SE ratio 0.6859 to 0.7183 | pass |
| `interval_calibration` | `in_sample_continuous__correctly_specified` | positive | law LB-c (normal outcome), in sample, the ATE, n = 2,000: one main-terms regression per arm, each correct: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9366 to 0.9515, SE ratio 0.9589 to 1.0029 | pass |
| `interval_calibration` | `in_sample_continuous__noise_control` | control | law LB-c (normal outcome), in sample, the ATE, n = 2,000: one main-terms regression per arm, each correct: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8176 to 0.8419, SE ratio 0.6819 to 0.7122 | pass |
| `interval_calibration` | `in_sample_continuous__shrunken_se_control` | control | law LB-c (normal outcome), in sample, the ATE, n = 2,000: one main-terms regression per arm, each correct: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8092 to 0.8340, SE ratio 0.6719 to 0.7021 | pass |
| `interval_calibration` | `stacked_binary__correctly_specified` | positive | law LB (binary outcome), stacked over five folds, the ATE, n = 2,000: one main-terms regression per arm, each correct: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9481 to 0.9616, SE ratio 0.9845 to 1.0302 | pass |
| `interval_calibration` | `stacked_binary__noise_control` | control | law LB (binary outcome), stacked over five folds, the ATE, n = 2,000: one main-terms regression per arm, each correct: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8288 to 0.8525, SE ratio 0.7007 to 0.7338 | pass |
| `interval_calibration` | `stacked_binary__shrunken_se_control` | control | law LB (binary outcome), stacked over five folds, the ATE, n = 2,000: one main-terms regression per arm, each correct: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8153 to 0.8398, SE ratio 0.6883 to 0.7210 | pass |
| `interval_calibration` | `stacked_bounded__correctly_specified` | positive | law LB-b (Beta outcome in (0, 1)), stacked over five folds, the ATE, n = 2,000: one main-terms regression per arm, each correct: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9436 to 0.9576, SE ratio 0.9810 to 1.0261 | pass |
| `interval_calibration` | `stacked_bounded__noise_control` | control | law LB-b (Beta outcome in (0, 1)), stacked over five folds, the ATE, n = 2,000: one main-terms regression per arm, each correct: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8174 to 0.8418, SE ratio 0.6859 to 0.7179 | pass |
| `interval_calibration` | `stacked_bounded__shrunken_se_control` | control | law LB-b (Beta outcome in (0, 1)), stacked over five folds, the ATE, n = 2,000: one main-terms regression per arm, each correct: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8136 to 0.8381, SE ratio 0.6867 to 0.7185 | pass |
| `pooled_outcome` | `in_sample_binary` | diagnostic | law LB (binary outcome), in sample, the ATE, n = 2,000: one pooled main-terms regression on (A, W1, W2), which misses the arm-by-covariate interaction, beside the correct treatment GLM, on the calibration cell's draws | none. The row is a diagnostic and states no verdict | bias -0.000404 to 0.0010, coverage 0.9527, SE ratio 1.0032, exact bias 0 SD, exact SE ratio 1.0001 | reported |
| `pooled_outcome` | `in_sample_continuous` | diagnostic | law LB-c (normal outcome), in sample, the ATE, n = 2,000: one pooled main-terms regression on (A, W1, W2), which misses the arm-by-covariate interaction, beside the correct treatment GLM, on the calibration cell's draws | none. The row is a diagnostic and states no verdict | bias -0.0031 to 0.000251, coverage 0.9442, SE ratio 0.9825 | reported |
| `pooled_outcome` | `stacked_binary` | diagnostic | law LB (binary outcome), stacked over five folds, the ATE, n = 2,000: one pooled main-terms regression on (A, W1, W2), which misses the arm-by-covariate interaction, beside the correct treatment GLM, on the calibration cell's draws | none. The row is a diagnostic and states no verdict | bias -0.000369 to 0.0010, coverage 0.9545, SE ratio 1.0074, exact bias 0 SD, exact SE ratio 1.0001 | reported |
| `pooled_outcome` | `stacked_bounded` | diagnostic | law LB-b (Beta outcome in (0, 1)), stacked over five folds, the ATE, n = 2,000: one pooled main-terms regression on (A, W1, W2), which misses the arm-by-covariate interaction, beside the correct treatment GLM, on the calibration cell's draws | none. The row is a diagnostic and states no verdict | bias -0.000221 to 0.000189, coverage 0.9516, SE ratio 1.0075, exact bias 0 SD, exact SE ratio 1.0010 | reported |
<!-- /generated -->

The four calibration cells read 6,400 draws each; the two binary cells share their draws. The
noise control adds one unit of the exact efficiency bound of the cell's law. The
double-robustness pair fits one intercept per arm for the outcome: with the correct treatment
GLM the bias is inside the margin, and with an intercept-only mechanism the bias is 3.34 SD
against the exact 3.43, so the pair can fail.

## Reported readings

The `pooled_outcome` rows fit one pooled main-terms regression on $(A, W_1, W_2)$ on each
calibration cell's draws. They carry no verdict and are not evidence that the per-arm fit is
better. Their exact limits are zero bias and an SE ratio of 1.0001 (binary) and 1.0010
(bounded); the continuous pooled fit has no fixed exact limit, because its fluctuation runs on
the sample range. The per-arm fit's efficiency gain over the pooled fit's limit is 0.19% on LB
and 2.36% on LB-b, too small for the study to resolve.

## Sizing

Every size was fixed before the run from the pilot in
`tests/studies/option_outcome_by_arm_pilot.py`, on a stream no declared cell reads. The
calibration count, 6,400, is the arm-pooled predictive rule at an 8,000-draw calibration pilot;
the spread between the binary and the continuous SE ratios raised it from the product floor of
2,800. The double-robustness cells run 1,600 and the primary phase 2,800. The design test
recomputes each count.

## Red-cell route

The study is registered under `reporting`, and no cell is red. The declared reading is in
`tests/canonical/option_outcome_by_arm/regenerate.py`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2800 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 3 | truth tests passing |
| `independent_tests_total` | 3 | truth tests reported |
| `property_cells_passed` | 14 | gated property cells passing |
| `property_cells_total` | 14 | gated property cells reported |
| `max_standardized_bias` | 0.0266 | largest primary standardized bias |
| `min_coverage` | 0.9454 | lowest primary coverage |
| `properties[interval_calibration/in_sample_binary__correctly_specified]:se_ratio` | 1.0025 | SE ratio, binary, in sample |
| `properties[interval_calibration/in_sample_continuous__correctly_specified]:se_ratio` | 0.9807 | SE ratio, continuous, in sample |
| `properties[interval_calibration/stacked_binary__correctly_specified]:se_ratio` | 1.0063 | SE ratio, binary, stacked |
| `properties[interval_calibration/stacked_bounded__correctly_specified]:se_ratio` | 1.0029 | SE ratio, bounded, stacked |
| `properties[interval_calibration/in_sample_continuous__correctly_specified]:coverage` | 0.9444 | lowest calibration coverage |
| `properties[double_robustness/both_wrong]:standardized_bias` | 3.3371 | standardized bias of the both-wrong control |
| `properties[double_robustness/both_wrong]:exact_bias_over_sd` | 3.4338 | its exact-law limit |
| `properties[pooled_outcome/stacked_bounded]:se_ratio` | 1.0075 | SE ratio of the pooled fit, bounded, stacked |
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
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 28 | the same bound, in primary replications |

## Limitations

- **The option's correctness rests on the fast tier.** No study cell would read differently if
  the option were ignored, so the exact-law witness and the spy carry that claim.
- **Main-terms GLMs per arm, two binary arms.** The cells measure correct parametric per-arm
  regressions on three laws with two normal covariates. Flexible per-arm learners are not
  measured.
- **No measured benefit.** The pooled fit is reported beside the per-arm fit and is unbiased
  under the correct mechanism here; the study claims the interval of the per-arm fit, not an
  advantage over the pooled one.
- **Arm means, the ATE, and their ratios.** The `T53` rows cover families P1 and P2. The ATT,
  ATC, population attributable quantities, DR-TMLE, C-TMLE, more than two arms and weights with
  the option read experimental.
- **No comparator.** No maintained implementation reports this interval.

## Reproduction

`tests/canonical/option_outcome_by_arm/regenerate.py` regenerates the study and states the run
form. It runs no reference process, because the study has no comparator. The manifest records
the seeds, the margins, the estimator configuration and the source and result hashes, and the
run log records the commit, the clean tree and the wall time.
