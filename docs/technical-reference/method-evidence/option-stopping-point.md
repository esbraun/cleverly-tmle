# Stopping rules on point TMLE

This study validates the interval of point `TMLE` under `stopping="scaled_var"` and
`stopping="sample_size"`, each on the Newton walk (`targeting="iterative"`) and on the one-step
walk. The [stopping rules section](../point-treatment-tmle.md#stopping-rules-step-bound-and-best-iterate)
states the rules. Under either rule the walk can stop before the score equation is solved, and
the estimate then carries the remaining score $r_n$ as a first-order term. The reported interval
adds it to the half-width, $\psi_n \pm (z\,\widehat{se} + |r_n|)$, which covers whenever the
plain Wald pivot does, whatever the residual's dependence on the error. The widened interval is
therefore conservative by construction, and the study measures how close to nominal it stays.

No package widens a Wald interval by its remaining score, so no canonical implementation is compared.
The study rests on exact truths and on repeated-sampling properties.

## What was fitted

| law | construction | fits |
| --- | --- | --- |
| L1 | the `canonical-tmle` binary and continuous laws, with that study's logistic (binary) or linear (continuous) outcome GLM and logistic treatment GLM | each rule on each walk, in sample, at n = 2,000; the ATE and the arm means |
| L2 | $W_1, W_2 \sim N(0, 1)$, $A \sim \operatorname{Bern}(\operatorname{expit}(0.4 W_1 - 0.3 W_2))$, $Y = 4 W_1 + 0.5 W_2 + 0.1 A + N(0, 1)$, with correct linear and logistic GLMs | the one-step walk under each rule; the ATE and the mean under treatment, plain and widened |
| LN | $V = W_1^2 - 1$ confounds treatment and outcome, $Y = W_1 + 0.5 W_2 + V + 0.1 A + 3 N(0, 1)$; the outcome regression omits $V$, so the initial score has a fixed sign | the one-step `scaled_var` walk with every threshold times 8, times 1, and under the relative rule; the mean under treatment |

Every fit uses `max_iter=100` (200 for the one-step walk), `tol=1e-10` and `canonical-tmle`'s
`g_bounds`. The laws, learners and the threshold scale are in
`tests/studies/_option_stopping_laws.py`, and
`tests/unit/test_option_stopping_point_design.py` recomputes every truth and declared number. The
fast-tier witnesses in `tests/unit/test_rule_widened_interval.py` check the widened interval,
the p-value and every consumer that reads it, each with a mutation that drops the residual.

The committed rows record the identity interval. The package now builds the interval of an
unclustered mean of a binary outcome on the logit scale
([small-sample scale](../inference.md#small-sample-scale)), and the residual enters there as
$\lvert r_n\rvert / (\psi_n(1 - \psi_n))$. The ATE cells read a difference, whose interval is
unchanged. On the same committed estimates, standard errors and residuals, the binary arm-mean row
covers 0.9603 against the recorded 0.9606, and no cell changes its verdict
(`tests/canonical/interval-audit-screen.csv`).

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE and the mean under treatment; stopping='scaled_var' on the one-step walk, the widened interval | `ate` | average treatment effect | `cleverly` | -0.000995 to 0.000915 | 0.9594 | 1.0068 | pass |
| the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE and the mean under treatment; stopping='scaled_var' on the one-step walk, the widened interval | `ey1` | counterfactual mean under treatment | `cleverly` | -0.000371 to 0.0010 | 0.9606 | 1.0119 | pass |
| the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE and the mean under treatment; stopping='scaled_var' on the one-step walk, the widened interval | `ate` | average treatment effect | `cleverly` | 0.000095 to 0.000523 | 0.9547 | 0.9963 | pass |
| the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE and the mean under treatment; stopping='scaled_var' on the one-step walk, the widened interval | `ey1` | counterfactual mean under treatment | `cleverly` | 0.000127 to 0.000657 | 0.9578 | 1.0008 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `contrast_residual` | `l2_ss_one_step_ate__plain` | diagnostic | law L2 (Y = 4 W1 + 0.5 W2 + 0.1 A + noise, correct linear and logistic GLMs), the ATE, n = 2,000: stopping='sample_size' on the one-step walk: the plain interval | none. The row is a diagnostic and states no verdict | coverage 0.9422 to 0.9513, SE ratio 0.9910, mean absolute residual 0.0039 SE | reported |
| `contrast_residual` | `l2_ss_one_step_ate__widened` | diagnostic | law L2 (Y = 4 W1 + 0.5 W2 + 0.1 A + noise, correct linear and logistic GLMs), the ATE, n = 2,000: stopping='sample_size' on the one-step walk: the widened interval | none. The row is a diagnostic and states no verdict | coverage 0.9429 to 0.9519, SE ratio 0.9910, mean absolute residual 0.0039 SE | reported |
| `contrast_residual` | `l2_ss_one_step_ey1__plain` | diagnostic | law L2 (Y = 4 W1 + 0.5 W2 + 0.1 A + noise, correct linear and logistic GLMs), the mean under treatment, n = 2,000: stopping='sample_size' on the one-step walk: the plain interval | none. The row is a diagnostic and states no verdict | coverage 0.9428 to 0.9518, SE ratio 0.9882, mean absolute residual 0.0018 SE | reported |
| `contrast_residual` | `l2_ss_one_step_ey1__widened` | diagnostic | law L2 (Y = 4 W1 + 0.5 W2 + 0.1 A + noise, correct linear and logistic GLMs), the mean under treatment, n = 2,000: stopping='sample_size' on the one-step walk: the widened interval | none. The row is a diagnostic and states no verdict | coverage 0.9431 to 0.9521, SE ratio 0.9882, mean absolute residual 0.0018 SE | reported |
| `contrast_residual` | `l2_sv_one_step_ate__plain` | diagnostic | law L2 (Y = 4 W1 + 0.5 W2 + 0.1 A + noise, correct linear and logistic GLMs), the ATE, n = 2,000: stopping='scaled_var' on the one-step walk: the plain interval | none. The row is a diagnostic and states no verdict | coverage 0.9437 to 0.9526, SE ratio 0.9946, mean absolute residual 0.0634 SE | reported |
| `contrast_residual` | `l2_sv_one_step_ate__widened` | diagnostic | law L2 (Y = 4 W1 + 0.5 W2 + 0.1 A + noise, correct linear and logistic GLMs), the ATE, n = 2,000: stopping='scaled_var' on the one-step walk: the widened interval | none. The row is a diagnostic and states no verdict | coverage 0.9505 to 0.9589, SE ratio 0.9946, mean absolute residual 0.0634 SE | reported |
| `contrast_residual` | `l2_sv_one_step_ey1__plain` | diagnostic | law L2 (Y = 4 W1 + 0.5 W2 + 0.1 A + noise, correct linear and logistic GLMs), the mean under treatment, n = 2,000: stopping='scaled_var' on the one-step walk: the plain interval | none. The row is a diagnostic and states no verdict | coverage 0.9426 to 0.9517, SE ratio 0.9902, mean absolute residual 0.0432 SE | reported |
| `contrast_residual` | `l2_sv_one_step_ey1__widened` | diagnostic | law L2 (Y = 4 W1 + 0.5 W2 + 0.1 A + noise, correct linear and logistic GLMs), the mean under treatment, n = 2,000: stopping='scaled_var' on the one-step walk: the widened interval | none. The row is a diagnostic and states no verdict | coverage 0.9471 to 0.9558, SE ratio 0.9902, mean absolute residual 0.0432 SE | reported |
| `interval_calibration` | `binary_ss_newton__noise_control` | control | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='sample_size' on the Newton walk: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8211 to 0.8364, SE ratio 0.6931 to 0.7133 | pass |
| `interval_calibration` | `binary_ss_newton__shrunken_se_control` | control | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='sample_size' on the Newton walk: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8242 to 0.8393, SE ratio 0.6917 to 0.7118 | pass |
| `interval_calibration` | `binary_ss_newton__widened` | positive | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='sample_size' on the Newton walk: the interval adds the remaining score to its half-width | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9444 to 0.9533, SE ratio 0.9884 to 1.0172 | pass |
| `interval_calibration` | `binary_ss_one_step__noise_control` | control | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='sample_size' on the one-step walk: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8230 to 0.8381, SE ratio 0.6913 to 0.7109 | pass |
| `interval_calibration` | `binary_ss_one_step__shrunken_se_control` | control | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='sample_size' on the one-step walk: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8246 to 0.8397, SE ratio 0.6921 to 0.7129 | pass |
| `interval_calibration` | `binary_ss_one_step__widened` | positive | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='sample_size' on the one-step walk: the interval adds the remaining score to its half-width | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9454 to 0.9542, SE ratio 0.9890 to 1.0173 | pass |
| `interval_calibration` | `binary_sv_newton__noise_control` | control | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='scaled_var' on the Newton walk: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8243 to 0.8394, SE ratio 0.6960 to 0.7159 | pass |
| `interval_calibration` | `binary_sv_newton__shrunken_se_control` | control | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='scaled_var' on the Newton walk: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8242 to 0.8393, SE ratio 0.6921 to 0.7123 | pass |
| `interval_calibration` | `binary_sv_newton__widened` | positive | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='scaled_var' on the Newton walk: the interval adds the remaining score to its half-width | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9444 to 0.9533, SE ratio 0.9883 to 1.0173 | pass |
| `interval_calibration` | `binary_sv_one_step__noise_control` | control | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='scaled_var' on the one-step walk: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8267 to 0.8417, SE ratio 0.6963 to 0.7164 | pass |
| `interval_calibration` | `binary_sv_one_step__shrunken_se_control` | control | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='scaled_var' on the one-step walk: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8242 to 0.8393, SE ratio 0.6932 to 0.7137 | pass |
| `interval_calibration` | `binary_sv_one_step__widened` | positive | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='scaled_var' on the one-step walk: the interval adds the remaining score to its half-width | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9496 to 0.9581, SE ratio 0.9904 to 1.0189 | pass |
| `interval_calibration` | `continuous_ss_newton__noise_control` | control | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='sample_size' on the Newton walk: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8278 to 0.8428, SE ratio 0.6986 to 0.7190 | pass |
| `interval_calibration` | `continuous_ss_newton__shrunken_se_control` | control | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='sample_size' on the Newton walk: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8228 to 0.8379, SE ratio 0.6925 to 0.7119 | pass |
| `interval_calibration` | `continuous_ss_newton__widened` | positive | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='sample_size' on the Newton walk: the interval adds the remaining score to its half-width | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9474 to 0.9561, SE ratio 0.9898 to 1.0175 | pass |
| `interval_calibration` | `continuous_ss_one_step__noise_control` | control | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='sample_size' on the one-step walk: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8250 to 0.8401, SE ratio 0.7006 to 0.7203 | pass |
| `interval_calibration` | `continuous_ss_one_step__shrunken_se_control` | control | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='sample_size' on the one-step walk: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8223 to 0.8374, SE ratio 0.6920 to 0.7119 | pass |
| `interval_calibration` | `continuous_ss_one_step__widened` | positive | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='sample_size' on the one-step walk: the interval adds the remaining score to its half-width | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9496 to 0.9581, SE ratio 0.9889 to 1.0171 | pass |
| `interval_calibration` | `continuous_sv_newton__noise_control` | control | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='scaled_var' on the Newton walk: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8265 to 0.8415, SE ratio 0.7012 to 0.7210 | pass |
| `interval_calibration` | `continuous_sv_newton__shrunken_se_control` | control | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='scaled_var' on the Newton walk: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8228 to 0.8379, SE ratio 0.6926 to 0.7118 | pass |
| `interval_calibration` | `continuous_sv_newton__widened` | positive | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='scaled_var' on the Newton walk: the interval adds the remaining score to its half-width | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9474 to 0.9561, SE ratio 0.9894 to 1.0176 | pass |
| `interval_calibration` | `continuous_sv_one_step__noise_control` | control | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='scaled_var' on the one-step walk: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8289 to 0.8438, SE ratio 0.7009 to 0.7210 | pass |
| `interval_calibration` | `continuous_sv_one_step__shrunken_se_control` | control | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='scaled_var' on the one-step walk: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8218 to 0.8370, SE ratio 0.6926 to 0.7123 | pass |
| `interval_calibration` | `continuous_sv_one_step__widened` | positive | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='scaled_var' on the one-step walk: the interval adds the remaining score to its half-width | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9531 to 0.9613, SE ratio 0.9896 to 1.0180 | pass |
| `necessity_reference` | `ln_ey1_k1__plain` | diagnostic | law LN (the outcome regression omits the confounding term V, so the initial score has a fixed sign), the mean under treatment, n = 2,000: one-step stopping='scaled_var' at the declared thresholds: the plain interval | none. The row is a diagnostic and states no verdict | coverage 0.9505 to 0.9589, SE ratio 1.0181, mean absolute residual 0.0595 SE | reported |
| `necessity_reference` | `ln_ey1_k1__widened` | diagnostic | law LN (the outcome regression omits the confounding term V, so the initial score has a fixed sign), the mean under treatment, n = 2,000: one-step stopping='scaled_var' at the declared thresholds: the widened interval | none. The row is a diagnostic and states no verdict | coverage 0.9562 to 0.9641, SE ratio 1.0181, mean absolute residual 0.0595 SE | reported |
| `necessity_reference` | `ln_ey1_rel__widened` | diagnostic | law LN (the outcome regression omits the confounding term V, so the initial score has a fixed sign), the mean under treatment, n = 2,000: one-step stopping='relative': the widened interval | none. The row is a diagnostic and states no verdict | coverage 0.9506 to 0.9590, SE ratio 1.0193, mean absolute residual 4.903e-09 SE | reported |
| `plain_interval` | `binary_ss_newton__plain` | diagnostic | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='sample_size' on the Newton walk: the plain Wald interval of the same fits, without the residual | none. The row is a diagnostic and states no verdict | coverage 0.9444 to 0.9533, SE ratio 1.0025, mean absolute residual 5.195e-12 SE | reported |
| `plain_interval` | `binary_ss_one_step__plain` | diagnostic | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='sample_size' on the one-step walk: the plain Wald interval of the same fits, without the residual | none. The row is a diagnostic and states no verdict | coverage 0.9444 to 0.9533, SE ratio 1.0031, mean absolute residual 0.0085 SE | reported |
| `plain_interval` | `binary_sv_newton__plain` | diagnostic | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='scaled_var' on the Newton walk: the plain Wald interval of the same fits, without the residual | none. The row is a diagnostic and states no verdict | coverage 0.9444 to 0.9533, SE ratio 1.0025, mean absolute residual 5.195e-12 SE | reported |
| `plain_interval` | `binary_sv_one_step__plain` | diagnostic | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='scaled_var' on the one-step walk: the plain Wald interval of the same fits, without the residual | none. The row is a diagnostic and states no verdict | coverage 0.9439 to 0.9529, SE ratio 1.0043, mean absolute residual 0.0417 SE | reported |
| `plain_interval` | `continuous_ss_newton__plain` | diagnostic | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='sample_size' on the Newton walk: the plain Wald interval of the same fits, without the residual | none. The row is a diagnostic and states no verdict | coverage 0.9474 to 0.9561, SE ratio 1.0032, mean absolute residual 0.000008 SE | reported |
| `plain_interval` | `continuous_ss_one_step__plain` | diagnostic | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='sample_size' on the one-step walk: the plain Wald interval of the same fits, without the residual | none. The row is a diagnostic and states no verdict | coverage 0.9467 to 0.9554, SE ratio 1.0029, mean absolute residual 0.0256 SE | reported |
| `plain_interval` | `continuous_sv_newton__plain` | diagnostic | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='scaled_var' on the Newton walk: the plain Wald interval of the same fits, without the residual | none. The row is a diagnostic and states no verdict | coverage 0.9474 to 0.9561, SE ratio 1.0032, mean absolute residual 0.000008 SE | reported |
| `plain_interval` | `continuous_sv_one_step__plain` | diagnostic | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='scaled_var' on the one-step walk: the plain Wald interval of the same fits, without the residual | none. The row is a diagnostic and states no verdict | coverage 0.9472 to 0.9559, SE ratio 1.0036, mean absolute residual 0.0537 SE | reported |
| `relative_reference` | `binary_rel_newton__widened` | diagnostic | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='relative' on the Newton walk: the default rule, whose residual is below tol | none. The row is a diagnostic and states no verdict | coverage 0.9444 to 0.9533, SE ratio 1.0025, mean absolute residual 5.195e-12 SE | reported |
| `relative_reference` | `continuous_rel_newton__widened` | diagnostic | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='relative' on the Newton walk: the default rule, whose residual is below tol | none. The row is a diagnostic and states no verdict | coverage 0.9474 to 0.9561, SE ratio 1.0032, mean absolute residual 1.893e-10 SE | reported |
| `residual_necessity` | `ln_ey1__plain` | control | law LN (the outcome regression omits the confounding term V, so the initial score has a fixed sign), the mean under treatment, n = 2,000: one-step stopping='scaled_var', every threshold times 8: the plain interval, without the residual | the upper limit of the coverage's exact 99% interval below 0.92 | coverage 0.8290 to 0.8440, SE ratio 1.0221, mean absolute residual 0.9828 SE | pass |
| `residual_necessity` | `ln_ey1__widened` | positive | law LN (the outcome regression omits the confounding term V, so the initial score has a fixed sign), the mean under treatment, n = 2,000: one-step stopping='scaled_var', every threshold times 8: the widened interval of the same fits | the lower limit of the coverage's exact 99% interval at least 0.90 | coverage 0.9742 to 0.9802, SE ratio 1.0221, mean absolute residual 0.9828 SE | pass |
| `residual_witness` | `binary_ss_newton__witness` | diagnostic | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='sample_size' on the Newton walk: the share of draws whose absolute residual is at least 0.01 standard errors | none. The row is a diagnostic and states no verdict | share 0 to 0.000323 | reported |
| `residual_witness` | `binary_ss_one_step__witness` | diagnostic | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='sample_size' on the one-step walk: the share of draws whose absolute residual is at least 0.01 standard errors | none. The row is a diagnostic and states no verdict | share 0.2938 to 0.3123 | reported |
| `residual_witness` | `binary_sv_newton__witness` | diagnostic | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='scaled_var' on the Newton walk: the share of draws whose absolute residual is at least 0.01 standard errors | none. The row is a diagnostic and states no verdict | share 0 to 0.000323 | reported |
| `residual_witness` | `binary_sv_one_step__witness` | positive | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='scaled_var' on the one-step walk: the share of draws whose absolute residual is at least 0.01 standard errors | the lower limit of the share's exact 99% interval at least 0.5 | share 0.8464 to 0.8607 | pass |
| `residual_witness` | `continuous_ss_newton__witness` | diagnostic | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='sample_size' on the Newton walk: the share of draws whose absolute residual is at least 0.01 standard errors | none. The row is a diagnostic and states no verdict | share 0 to 0.000323 | reported |
| `residual_witness` | `continuous_ss_one_step__witness` | diagnostic | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='sample_size' on the one-step walk: the share of draws whose absolute residual is at least 0.01 standard errors | none. The row is a diagnostic and states no verdict | share 0.7420 to 0.7594 | reported |
| `residual_witness` | `continuous_sv_newton__witness` | diagnostic | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='scaled_var' on the Newton walk: the share of draws whose absolute residual is at least 0.01 standard errors | none. The row is a diagnostic and states no verdict | share 0 to 0.000323 | reported |
| `residual_witness` | `continuous_sv_one_step__witness` | positive | the canonical-tmle continuous law, a linear outcome and a logistic treatment GLM, in sample, n = 2,000, the ATE: stopping='scaled_var' on the one-step walk: the share of draws whose absolute residual is at least 0.01 standard errors | the lower limit of the share's exact 99% interval at least 0.5 | share 0.9979 to 0.9994 | pass |
| `step_bound` | `binary_ss_newton_step_bound__widened` | diagnostic | the canonical-tmle binary law, logistic outcome and treatment GLMs, in sample, n = 2,000, the ATE: stopping='sample_size' on the Newton walk with step_bound=0.05, the tmle3 pairing: the widened interval | none. The row is a diagnostic and states no verdict | coverage 0.9450 to 0.9539, SE ratio 1.0026, mean absolute residual 0.0041 SE | reported |
<!-- /generated -->

The eight calibration positives read the widened interval of the ATE on L1. On the Newton walk
the rule never binds: the largest residual is $10^{-10}$ SE on the binary law and 0.0014 SE on
the continuous law, no draw reaches 0.01 SE, and coverage and SE ratio equal the relative fit's
on the same draws. Those four cells measure the default estimator under a rule. On the one-step
walk `scaled_var` leaves a residual of at least 0.01 SE on 85% (binary) and 99.9% (continuous)
of draws, and `sample_size` on 30% and 75%. The widened interval covers 0.950 to 0.957 there.

The necessity pair fits LN with every threshold times 8, so the walk stops near a residual of
eight times the rule's cap. The plain interval covers 0.837 and the widened interval 0.977 on
the same fits, so the widening is what keeps coverage.

## Reported readings

These rows carry no verdict. The plain interval of every calibration fit covers 0.949 to 0.952,
within 0.006 of the widened one, because the residual is small against the standard error on
L1. At the declared thresholds LN covers 0.955 plain and 0.960 widened, and the relative fit
0.955. On L2, `scaled_var` widens the ATE's coverage from 0.948 to 0.955. The `sample_size`
Newton walk with `step_bound=0.05`, the `tmle3` pairing, covers 0.950; the support key keeps it
experimental.

## Sizing

Every size was fixed before the run from the 2,000-draw pilot in
`tests/studies/option_stopping_pilot.py`, on a stream no declared cell reads. Every property
cell runs 16,400 replications: the product floor is 3,200, and the arm-pooled predictive rule
raises it, because the cells of one law read the same draws. The primary phase runs 3,200. The
design test recomputes each count.

## Red-cell route

The study is registered under `reporting`, and no cell is red. The declared reading is in
`tests/canonical/option_stopping_point/regenerate.py`, and
`tests/unit/test_option_stopping_point_design.py` applies it to the committed tables through
`gating_findings`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 3200 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 4 | truth tests passing |
| `independent_tests_total` | 4 | truth tests reported |
| `property_cells_passed` | 28 | gated property cells passing |
| `property_cells_total` | 28 | gated property cells reported |
| `max_standardized_bias` | 0.0674 | largest primary standardized bias |
| `min_coverage` | 0.9547 | lowest primary coverage |
| `properties[interval_calibration/binary_sv_one_step__widened]:coverage` | 0.9540 | widened coverage, binary law, scaled_var, one-step |
| `properties[interval_calibration/continuous_sv_one_step__widened]:coverage` | 0.9573 | widened coverage, continuous law, scaled_var, one-step |
| `properties[interval_calibration/binary_ss_newton__widened]:coverage` | 0.9490 | widened coverage, binary law, sample_size, Newton |
| `properties[relative_reference/binary_rel_newton__widened]:coverage` | 0.9490 | the relative fit on the same draws |
| `properties[residual_witness/binary_sv_one_step__witness]:coverage` | 0.8537 | share of draws with a residual of at least 0.01 SE, binary, scaled_var, one-step |
| `properties[residual_witness/continuous_sv_one_step__witness]:coverage` | 0.9988 | the same share, continuous law |
| `properties[residual_witness/binary_ss_one_step__witness]:coverage` | 0.3030 | the same share, binary law, sample_size |
| `properties[residual_witness/continuous_ss_one_step__witness]:coverage` | 0.7508 | the same share, continuous law, sample_size |
| `properties[residual_necessity/ln_ey1__plain]:coverage` | 0.8366 | plain coverage of the necessity fits |
| `properties[residual_necessity/ln_ey1__widened]:coverage` | 0.9773 | widened coverage of the necessity fits |
| `properties[step_bound/binary_ss_newton_step_bound__widened]:coverage` | 0.9496 | the step-bounded Newton walk |
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
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 32 | the same bound, in primary replications |

## Limitations

- **The widened interval is conservative.** Its coverage rests on the plain pivot alone, so
  where the rule binds hard it over-covers (0.977 in the necessity cell). No cell gates an
  efficiency claim.
- **The Newton cells measure the default estimator.** On L1 the Newton walk lands at the
  relative fit, so those cells show no rule effect and license only that walk.
- **Two laws, in sample.** The calibration cells read the ATE on the `canonical-tmle` binary and
  continuous laws at n = 2,000 with parametric learners. Cross-fitting, other families and other
  learners read experimental through the support key.
- **A step bound or `iterate="best"` under a rule is not covered.** The key records them and no
  row declares them.
- **The bootstrap is refused** under a rule, because no source bounds what the residual does to
  a refit's law.
- **No comparator.** No package widens a Wald interval by its remaining score.
- **The binary rows read today's normal-quantile interval.** The binary arm means are built on
  the logit scale with $z\,\widehat{\mathrm{se}}$. The planned small-sample interval for bounded
  estimates changes that half-width, so the binary-outcome interval rows will be recomputed with
  the composed half-width, the small-sample half-width plus $\lvert r_n\rvert$, when it ships.

## Reproduction

`tests/canonical/option_stopping_point/regenerate.py` regenerates the study and states the run
form. It runs no reference process, because the study has no comparator. The manifest records
the seeds, the margins, the estimator configuration and the source and result hashes, and the
run log records the commit, the clean tree and the wall time.
