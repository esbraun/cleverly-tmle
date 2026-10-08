# Point-treatment survival policies

This study validates longitudinal TMLE on the held survival design under a policy at the one
treatment decision. A known stochastic policy or a modified treatment policy sets the treatment
at node 1, and identity nodes follow it. That is Díaz, Williams, Hoffman and Schenck (2023),
Definition 1, with $d_t(a_t, h_t) = a_t$ for $t \ge 2$. The section
[one baseline treatment held over the nodes](../longitudinal-tmle.md#one-baseline-treatment-held-over-the-nodes)
states the estimator and its exact-law evidence.

The study pairs each fit with R `lmtp` 1.5.4. In `lmtp` the held policy is `trt` of length K.
That is K copied treatment columns. The first holds the policy value, and the others hold the
observed value. With `trt` of length one, `lmtp` applies the policy once at every node. For a
policy that is not idempotent, that is a different estimand.

## What was compared

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets | 1,600 samples of 2,000 rows for each scenario | the identical rows, with K copied treatment columns |
| scenarios | `policy`: the five-visit binary law, with $q(1 \mid W) = 0.5 + 0.25 W_1 + 0.25 \cdot 1\{W_2 \ge 2\}$ against the natural course. `mtp`: the dose law on `0..5`, with `lmtp`'s Example 2.1 policy `a - 1` where `a - 1 >= 1`, against the natural course | the same |
| reported parameters | both risks and their difference at visits 1, 3 and 5 | the same |
| mechanism | the law's own treatment and retention probabilities | the density ratio from the same probabilities, written beside the replicate data. `lmtp` fits no ratio |
| sequential regressions | a quasibinomial regression on `W1`, `W2` and the treatment | `SL.glm` on the same design, with the dose as a factor |
| targeting | the intercept of each node, with the clever covariate in the loss weight | the intercept fluctuation weighted by the ratio |
| intervals | pointwise 95% Wald from the influence curve | the same, from the `lmtp` influence curve |

The two fluctuations differ by construction, so the paired rows read under the default margins.
The `initial_estimate` column is one quantity on the two sides: the node-1 regression at the
policy treatment, fitted on pseudo-outcomes that the later nodes already targeted. No verdict
reads it.

Replication 1,003 of `mtp` has two sequential regressions with quasi-complete separation. The
quasibinomial learner keeps the last iterate with a warning, as R's `glm` does.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `ate_regimen[baseline minus one vs natural course @ t=1]` | difference in cumulative risk between the plans "lower the baseline dose by one wherever it stays at least one, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000138 to 0.000554 | 0.9450 | 0.9983 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `ate_regimen[baseline minus one vs natural course @ t=1]` | difference in cumulative risk between the plans "lower the baseline dose by one wherever it stays at least one, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | R `lmtp` | -0.000138 to 0.000554 | 0.9450 | 0.9983 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `ate_regimen[baseline minus one vs natural course @ t=3]` | difference in cumulative risk between the plans "lower the baseline dose by one wherever it stays at least one, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000222 to 0.000846 | 0.9419 | 0.9873 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `ate_regimen[baseline minus one vs natural course @ t=3]` | difference in cumulative risk between the plans "lower the baseline dose by one wherever it stays at least one, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | R `lmtp` | -0.000224 to 0.000842 | 0.9425 | 0.9880 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `ate_regimen[baseline minus one vs natural course @ t=5]` | difference in cumulative risk between the plans "lower the baseline dose by one wherever it stays at least one, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 5 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000319 to 0.000850 | 0.9519 | 1.0085 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `ate_regimen[baseline minus one vs natural course @ t=5]` | difference in cumulative risk between the plans "lower the baseline dose by one wherever it stays at least one, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 5 | R `lmtp` | -0.000319 to 0.000850 | 0.9519 | 1.0087 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[baseline minus one @ t=1]` | cumulative risk under the plan lower the baseline dose by one wherever it stays at least one, then hold it at horizon t = 1 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000360 to 0.000752 | 0.9544 | 1.0419 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[baseline minus one @ t=1]` | cumulative risk under the plan lower the baseline dose by one wherever it stays at least one, then hold it at horizon t = 1 | R `lmtp` | -0.000360 to 0.000752 | 0.9544 | 1.0419 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[baseline minus one @ t=3]` | cumulative risk under the plan lower the baseline dose by one wherever it stays at least one, then hold it at horizon t = 3 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000734 to 0.000973 | 0.9581 | 1.0309 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[baseline minus one @ t=3]` | cumulative risk under the plan lower the baseline dose by one wherever it stays at least one, then hold it at horizon t = 3 | R `lmtp` | -0.000737 to 0.000969 | 0.9575 | 1.0311 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[baseline minus one @ t=5]` | cumulative risk under the plan lower the baseline dose by one wherever it stays at least one, then hold it at horizon t = 5 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000325 to 0.0016 | 0.9469 | 0.9937 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[baseline minus one @ t=5]` | cumulative risk under the plan lower the baseline dose by one wherever it stays at least one, then hold it at horizon t = 5 | R `lmtp` | -0.000322 to 0.0016 | 0.9469 | 0.9938 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[natural course @ t=1]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 1 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000433 to 0.000409 | 0.9600 | 1.0320 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[natural course @ t=1]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 1 | R `lmtp` | -0.000433 to 0.000409 | 0.9600 | 1.0320 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[natural course @ t=3]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 3 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000858 to 0.000471 | 0.9556 | 1.0239 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[natural course @ t=3]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 3 | R `lmtp` | -0.000858 to 0.000471 | 0.9556 | 1.0238 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[natural course @ t=5]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 5 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000410 to 0.0012 | 0.9400 | 0.9725 | pass |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[natural course @ t=5]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 5 | R `lmtp` | -0.000407 to 0.0012 | 0.9400 | 0.9725 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `ate_regimen[baseline policy vs natural course @ t=1]` | difference in cumulative risk between the plans "draw the baseline arm from a known policy, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000287 to 0.000267 | 0.9425 | 0.9623 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `ate_regimen[baseline policy vs natural course @ t=1]` | difference in cumulative risk between the plans "draw the baseline arm from a known policy, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | R `lmtp` | -0.000289 to 0.000266 | 0.9419 | 0.9610 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `ate_regimen[baseline policy vs natural course @ t=3]` | difference in cumulative risk between the plans "draw the baseline arm from a known policy, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000605 to 0.000195 | 0.9519 | 1.0070 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `ate_regimen[baseline policy vs natural course @ t=3]` | difference in cumulative risk between the plans "draw the baseline arm from a known policy, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | R `lmtp` | -0.000615 to 0.000186 | 0.9525 | 1.0066 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `ate_regimen[baseline policy vs natural course @ t=5]` | difference in cumulative risk between the plans "draw the baseline arm from a known policy, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 5 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000497 to 0.000385 | 0.9556 | 1.0092 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `ate_regimen[baseline policy vs natural course @ t=5]` | difference in cumulative risk between the plans "draw the baseline arm from a known policy, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 5 | R `lmtp` | -0.000501 to 0.000381 | 0.9563 | 1.0089 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[baseline policy @ t=1]` | cumulative risk under the plan draw the baseline arm from a known policy, then hold it at horizon t = 1 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000261 to 0.000729 | 0.9400 | 0.9794 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[baseline policy @ t=1]` | cumulative risk under the plan draw the baseline arm from a known policy, then hold it at horizon t = 1 | R `lmtp` | -0.000263 to 0.000728 | 0.9400 | 0.9792 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[baseline policy @ t=3]` | cumulative risk under the plan draw the baseline arm from a known policy, then hold it at horizon t = 3 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000513 to 0.0010 | 0.9531 | 1.0145 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[baseline policy @ t=3]` | cumulative risk under the plan draw the baseline arm from a known policy, then hold it at horizon t = 3 | R `lmtp` | -0.000522 to 0.000995 | 0.9525 | 1.0149 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[baseline policy @ t=5]` | cumulative risk under the plan draw the baseline arm from a known policy, then hold it at horizon t = 5 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000371 to 0.0014 | 0.9575 | 1.0360 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[baseline policy @ t=5]` | cumulative risk under the plan draw the baseline arm from a known policy, then hold it at horizon t = 5 | R `lmtp` | -0.000374 to 0.0014 | 0.9563 | 1.0358 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[natural course @ t=1]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 1 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000210 to 0.000697 | 0.9481 | 0.9965 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[natural course @ t=1]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 1 | R `lmtp` | -0.000210 to 0.000697 | 0.9481 | 0.9965 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[natural course @ t=3]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 3 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000229 to 0.0011 | 0.9531 | 1.0225 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[natural course @ t=3]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 3 | R `lmtp` | -0.000229 to 0.0011 | 0.9537 | 1.0224 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[natural course @ t=5]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 5 | `cleverly` LTMLE with a policy at a held baseline treatment | -0.000203 to 0.0013 | 0.9619 | 1.0366 | pass |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[natural course @ t=5]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 5 | R `lmtp` | -0.000202 to 0.0013 | 0.9625 | 1.0366 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `ate_regimen[baseline minus one vs natural course @ t=1]` | difference in cumulative risk between the plans "lower the baseline dose by one wherever it stays at least one, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | 3.518e-10 | 4.374e-07 | 1.0000 | 0 | 8.393e-09 vs 0.0500 | equivalent |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `ate_regimen[baseline minus one vs natural course @ t=3]` | difference in cumulative risk between the plans "lower the baseline dose by one wherever it stays at least one, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | 0.000003 | 0.0024 | 1.0018 | -0.000625 | 0.000948 vs 0.0500 | equivalent |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `ate_regimen[baseline minus one vs natural course @ t=5]` | difference in cumulative risk between the plans "lower the baseline dose by one wherever it stays at least one, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 5 | -1.421e-07 | 0.000104 | 1.0007 | 0 | 0.000715 vs 0.0500 | equivalent |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[baseline minus one @ t=1]` | cumulative risk under the plan lower the baseline dose by one wherever it stays at least one, then hold it at horizon t = 1 | -1.660e-10 | 1.283e-07 | 1.0000 | 0 | 3.806e-09 vs 0.0500 | equivalent |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[baseline minus one @ t=3]` | cumulative risk under the plan lower the baseline dose by one wherever it stays at least one, then hold it at horizon t = 3 | 0.000003 | 0.0017 | 1.0009 | 0.000625 | 0.000757 vs 0.0500 | equivalent |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[baseline minus one @ t=5]` | cumulative risk under the plan lower the baseline dose by one wherever it stays at least one, then hold it at horizon t = 5 | -0.000003 | 0.0012 | 1.0004 | 0 | 0.000275 vs 0.0500 | equivalent |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[natural course @ t=1]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 1 | -5.178e-10 | 5.286e-07 | 1.0000 | 0 | 7.155e-09 vs 0.0500 | equivalent |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[natural course @ t=3]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 3 | 3.022e-07 | 0.000195 | 1.0001 | 0 | 0.000191 vs 0.0500 | equivalent |
| five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline | `risk_regimen[natural course @ t=5]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 5 | -0.000003 | 0.0013 | 1.0002 | 0 | 0.000224 vs 0.0500 | equivalent |
| five visits, a held binary treatment, a known stochastic policy at baseline | `ate_regimen[baseline policy vs natural course @ t=1]` | difference in cumulative risk between the plans "draw the baseline arm from a known policy, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | 0.000002 | 0.0024 | 1.0002 | 0.000625 | 0.0023 vs 0.0500 | equivalent |
| five visits, a held binary treatment, a known stochastic policy at baseline | `ate_regimen[baseline policy vs natural course @ t=3]` | difference in cumulative risk between the plans "draw the baseline arm from a known policy, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | 0.000009 | 0.0097 | 1.0008 | -0.000625 | 0.0019 vs 0.0500 | equivalent |
| five visits, a held binary treatment, a known stochastic policy at baseline | `ate_regimen[baseline policy vs natural course @ t=5]` | difference in cumulative risk between the plans "draw the baseline arm from a known policy, then hold it" against "leave the observed treatment mechanism unchanged" at horizon t = 5 | 0.000004 | 0.0037 | 1.0011 | -0.000625 | 0.0015 vs 0.0500 | equivalent |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[baseline policy @ t=1]` | cumulative risk under the plan draw the baseline arm from a known policy, then hold it at horizon t = 1 | 0.000002 | 0.0013 | 1.0003 | 0 | 0.000695 vs 0.0500 | equivalent |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[baseline policy @ t=3]` | cumulative risk under the plan draw the baseline arm from a known policy, then hold it at horizon t = 3 | 0.000009 | 0.0052 | 1.0013 | 0.000625 | 0.0015 vs 0.0500 | equivalent |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[baseline policy @ t=5]` | cumulative risk under the plan draw the baseline arm from a known policy, then hold it at horizon t = 5 | 0.000002 | 0.0012 | 1.0005 | 0.0012 | 0.000887 vs 0.0500 | equivalent |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[natural course @ t=1]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 1 | -5.574e-10 | 5.281e-07 | 1.0000 | 0 | 3.734e-09 vs 0.0500 | equivalent |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[natural course @ t=3]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 3 | 1.947e-07 | 0.000123 | 1.0001 | -0.000625 | 0.000212 vs 0.0500 | equivalent |
| five visits, a held binary treatment, a known stochastic policy at baseline | `risk_regimen[natural course @ t=5]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 5 | -0.000001 | 0.000813 | 1.0002 | -0.000625 | 0.000258 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `mtp_t5__correctly_specified` | positive | baseline minus one against the natural course at visit 5: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9406 to 0.9678, SE ratio 0.9810 to 1.0681, empirical efficiency ratio 0.9732 to 1.0603, reported efficiency ratio 1.0378 to 1.0431 | pass |
| `interval_calibration` | `mtp_t5__noise_control` | control | baseline minus one against the natural course at visit 5: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8177 to 0.8652, SE ratio 0.6959 to 0.7635, empirical efficiency ratio 1.3630 to 1.4952, reported efficiency ratio 1.0378 to 1.0430 | pass |
| `interval_calibration` | `mtp_t5__shrunken_se_control` | control | baseline minus one against the natural course at visit 5: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8072 to 0.8558, SE ratio 0.6860 to 0.7473, empirical efficiency ratio 0.9743 to 1.0614, reported efficiency ratio 0.7264 to 0.7300 | pass |
| `interval_calibration` | `policy_t5__correctly_specified` | positive | baseline policy against the natural course at visit 5: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9328 to 0.9619, SE ratio 0.9415 to 1.0350, empirical efficiency ratio 0.9775 to 1.0742, reported efficiency ratio 1.0094 to 1.0143 | pass |
| `interval_calibration` | `policy_t5__noise_control` | control | baseline policy against the natural course at visit 5: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8171 to 0.8646, SE ratio 0.6737 to 0.7394, empirical efficiency ratio 1.3689 to 1.5018, reported efficiency ratio 1.0094 to 1.0143 | pass |
| `interval_calibration` | `policy_t5__shrunken_se_control` | control | baseline policy against the natural course at visit 5: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8006 to 0.8500, SE ratio 0.6595 to 0.7230, empirical efficiency ratio 0.9793 to 1.0740, reported efficiency ratio 0.7065 to 0.7099 | pass |
<!-- /generated -->

The positive cells fit the saturated cell means for every nuisance, so the mechanism and the ratio
numerator are estimated. Each efficiency bound is the exact standard deviation of the efficient
influence function over the law's support. For the natural course and the modified treatment
policy, the treatment mechanism is part of the parameter, and the bound carries its term.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications per scenario |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 36 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 36 | of those, passing |
| `paired_tests_total` | 18 | paired comparisons with `lmtp` |
| `paired_tests_passed` | 18 | of those, passing |
| `property_cells_total` | 6 | repeated-sampling property cells |
| `property_cells_passed` | 6 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0472 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9400 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9202 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0941 | highest bootstrap primary SE-ratio endpoint |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed |

## Limitations

| limit | what it means for use |
| --- | --- |
| a policy at node 1 only | a plan that changes the treatment after baseline is refused on a held design |
| the mechanism is the law's own in the paired fits | both sides read the same probabilities, so the pairs test the targeting and the recursion, not the mechanism fit |
| the paired rows are not an exactness check | the two fluctuations differ by construction |
| one categorical dose law | a continuous baseline dose has fast-tier evidence only |
| calibration only | the property cells measure interval calibration. Double robustness and rates for the held design are in [point-treatment survival](point-treatment-survival.md) |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_policies/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. Run `python -m tests.canonical.point_survival_policies.regenerate` to regenerate the
artifacts. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_policies/replicates.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_policies/properties.csv)
carry every published row.
