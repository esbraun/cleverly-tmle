# Sampled refutation verdict rules

This study checks `refute(result, sample_limit=m)`, which refutes a sample of a fit's units
instead of all of them. The call draws `m` of the fit's `n` independent units without replacement,
refits the stored estimator on them, and runs every test on that sampled baseline as on any fit of
`m` units. The [sampled refutation](../validation-methods.md#refuting-a-sample-of-the-units)
section of the validation reference defines the call.

Under i.i.d. units the sampled verdict has exactly the distribution of the full-data verdict at
`n = m`, whatever `m/n`. The argument is exact: a simple random sample of `m` i.i.d. units is
itself an i.i.d. sample of size `m`, and the refit, the tests and their seeds read only those
units. The same holds for whole clusters when the clusters are the i.i.d. unit. The study reads
the claim by `m` and `m/n`, i.i.d. and in whole clusters, against the full-data rule at `n = m`.

| rule | reading | rejects when |
| --- | --- | --- |
| placebo `t` | 20 permutation placebo refits against the exact null, `t` with 19 degrees of freedom | the two-sided p-value is below 0.05 |
| subset `rho` | 20 subset refits against the sampling reference of the baseline's standard error | the two-sided $\chi^2_{19}$ p-value is below 0.05 |
| negative-control `z` | one refit on a declared negative-control outcome | the two-sided p-value is below 0.05 |

The study adds no verdict row. A sampled call reads the existing verdict rows at its sampled
baseline's configuration, so a sample of 2,000 units from the T1 configuration of the
[point-treatment refutation study](refuter-calibration-point.md) prints a verdict, and a sample of
1,000 units, below that row's `n_min`, prints its numbers and no verdict. No verdict row has a
clustered scope, so a clustered sample prints no verdict either.

**No canonical implementation is compared.** No maintained implementation refutes a sample of a
fit's units. The zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| law and fit | the T1 cell of the point-treatment refutation study: 20 standard normal covariates, an unmeasured $U$ in the treatment and the confounded control only, $Y = A + 0.5 W_1 + 0.4 W_2 + 0.05 (W_3 + \dots + W_{20}) + e$, and `TMLE` of the ATE with linear and logistic GLM nuisances at five folds. The ATE is 1 |
| clustered law | rows in clusters of five. A cluster effect $b_Y \sim N(0, 0.5^2)$ enters the outcome and both controls, an independent $b_A$ from the same law enters the treatment logit, and an independent modifier $c \sim N(0, 1)$ gives $Y = (1 + c) A + L(W) + b_Y + e$. The effects confound nothing, so the ATE stays 1. The fit declares the clusters, so its inference is cluster-robust and its folds keep clusters whole |
| I2 cells | $m = 2000$ subjects at $n = 2000$ (I2-F, a limit that does not bind), 4,000 (I2-H) and 20,000 (I2-T). I2-F is the registered T1 configuration |
| I1 cells | $m = 1000$ subjects at $n = 1000$ (I1-F) and 10,000 (I1-T), below the verdict row's `n_min` |
| clustered cells | $m = 400$ clusters at 400 (C-F), 800 (C-H) and 4,000 (C-T) clusters |
| P cells | the full rule at $n = 4000$ and 20,000, 50 trials each, reported for power |
| trials | 1,000 per I and C cell. Each trial fits once at $n$ and calls `refute` with `random_state=None`, so the sample and refit seeds derive from the fit's seed |
| defects | trials 0 to 49 of every cell at two levels. The strong level is the point study's: 5% of rows keep their treatment in the placebo, the subset reading's SE halved and doubled, and a control confounded by $U$ at 0.45. The weak level (2.5%, factors 0.75 and 1.43, and 0.15) is sized for a per-trial power near 0.5 at $m = 2000$ |
| absolute size | `refuter_size`, the I2 cells' three rules: at most 73 rejections of 1,000 |
| relative size | `refuter_size_relative`, each sampled cell against its full cell: the 90% Wald interval of the difference of rejection rates lies inside $\pm 0.05$ |
| power | `refuter_power`, the I2 cells' strong defects: at least 38 detections of 50 |
| mutation | `refuter_mutation`, I2-T, I1-T and C-T: the subset `rho` recomputed with the full fit's SE and unit count, which is what an implementation that sliced the full fit would read, must fail both the absolute count and the relative test |
| primary phase | 200 replications of the I2-F ATE against its exact truth, as a sanity check of the fit |
| Monte Carlo inference | exact 99% intervals around every rate, and 99% intervals around every primary endpoint |

The X106 part H comparison plan fixed the laws, the cells, the gates, the seeds and the red-cell
route before the run. The module docstring of `tests/studies/refuter_sampled.py` states each law
and defect. `tests/unit/test_refuter_sampled_design.py` sizes each defect from the law, checks that
the clustered design effect is at least 1.4, and computes the gates' operating characteristics.

With every rule at size 0.05 and every strong defect at power 0.90, the chance of a false red in
any of the 39 gated readings is at most 0.03. That bound needs the size to be 0.05: a rule of true
size 0.06 would turn one of the nine absolute readings red with probability about 0.30.

## Accuracy against known truth

The primary phase checks the fit that the rules read. It does not calibrate a rule.

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the point refutation study's linear Gaussian law at n = 2,000; TMLE at five folds, refuted on its whole data | `ate` | average treatment effect | `cleverly` fit that the sampled refutation rules read | 0.000158 to 0.0149 | 0.9800 | 1.1968 | **fail** |
<!-- /generated -->

## Theory properties

A `refuter_size` row counts rejections of a rule under its null. A `refuter_size_relative` row
compares a sampled cell's rejection rate with its full cell's, and its `__partner` twin publishes
the full cell's rate with no verdict. A `refuter_power` row counts detections of a strong defect
on the first 50 trials, and a `refuter_mutation` row reads the sliced-fit reference. A
`refuter_reported` row is a diagnostic: the I1, clustered and P cells' rates and every weak defect.
The cell name gives the configuration, then the reading.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `refuter_mutation` | `C-T__subset_full_reference` | control | 4,000 clusters of five and a sample of m = 400 whole clusters: the subset rule read against the full-data fit, the mutation that skips the sampled baseline | the rejection count exceeds 73 of 1,000, and the 90% interval of its rate minus the full cell's subset rate lies wholly outside -0.05 to 0.05 | rejections 881 of 1000, difference 0.8054 to 0.8466 | pass |
| `refuter_mutation` | `I1-T__subset_full_reference` | control | independent rows, n = 10,000 and a sample of m = 1,000: the subset rule read against the full-data fit, the mutation that skips the sampled baseline | the rejection count exceeds 73 of 1,000, and the 90% interval of its rate minus the full cell's subset rate lies wholly outside -0.05 to 0.05 | rejections 793 of 1000, difference 0.7233 to 0.7707 | pass |
| `refuter_mutation` | `I2-T__subset_full_reference` | control | independent rows, n = 20,000 and a sample of m = 2,000: the subset rule read against the full-data fit, the mutation that skips the sampled baseline | the rejection count exceeds 73 of 1,000, and the 90% interval of its rate minus the full cell's subset rate lies wholly outside -0.05 to 0.05 | rejections 865 of 1000, difference 0.7897 to 0.8323 | pass |
| `refuter_power` | `I2-F__control_confounded` | control | independent rows, n = 2,000 and m = 2,000 (a limit that does not bind): the negative-control rule, on a control confounded by U at 0.45 | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `I2-F__placebo_stale_rows` | control | independent rows, n = 2,000 and m = 2,000 (a limit that does not bind): the placebo rule, with 5% of rows keeping their treatment | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `I2-F__subset_double_se` | control | independent rows, n = 2,000 and m = 2,000 (a limit that does not bind): the subset rule, at the sampled baseline's SE doubled | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `I2-F__subset_half_se` | control | independent rows, n = 2,000 and m = 2,000 (a limit that does not bind): the subset rule, at the sampled baseline's SE halved | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `I2-H__control_confounded` | control | independent rows, n = 4,000 and a sample of m = 2,000: the negative-control rule, on a control confounded by U at 0.45 | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `I2-H__placebo_stale_rows` | control | independent rows, n = 4,000 and a sample of m = 2,000: the placebo rule, with 5% of rows keeping their treatment | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `I2-H__subset_double_se` | control | independent rows, n = 4,000 and a sample of m = 2,000: the subset rule, at the sampled baseline's SE doubled | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `I2-H__subset_half_se` | control | independent rows, n = 4,000 and a sample of m = 2,000: the subset rule, at the sampled baseline's SE halved | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `I2-T__control_confounded` | control | independent rows, n = 20,000 and a sample of m = 2,000: the negative-control rule, on a control confounded by U at 0.45 | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `I2-T__placebo_stale_rows` | control | independent rows, n = 20,000 and a sample of m = 2,000: the placebo rule, with 5% of rows keeping their treatment | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `I2-T__subset_double_se` | control | independent rows, n = 20,000 and a sample of m = 2,000: the subset rule, at the sampled baseline's SE doubled | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `I2-T__subset_half_se` | control | independent rows, n = 20,000 and a sample of m = 2,000: the subset rule, at the sampled baseline's SE halved | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_reported` | `C-F__control_confounded` | diagnostic | 400 clusters of five, with m = 400 clusters (a limit that does not bind): the rejection rate of the negative-control rule, on a control confounded by U at 0.45 | none. The row is a diagnostic and states no verdict | rejection 0.9200, 0.7689 to 0.9862 | reported |
| `refuter_reported` | `C-F__control_confounded_weak` | diagnostic | 400 clusters of five, with m = 400 clusters (a limit that does not bind): the rejection rate of the negative-control rule, on a control confounded by U at 0.15 | none. The row is a diagnostic and states no verdict | rejection 0.3800, 0.2121 to 0.5713 | reported |
| `refuter_reported` | `C-F__negative_control` | diagnostic | 400 clusters of five, with m = 400 clusters (a limit that does not bind): the rejection rate of the negative-control rule on an outcome the treatment does not move | none. The row is a diagnostic and states no verdict | rejection 0.0350, 0.0218 to 0.0528 | reported |
| `refuter_reported` | `C-F__placebo` | diagnostic | 400 clusters of five, with m = 400 clusters (a limit that does not bind): the rejection rate of the placebo t rule against the exact null 0 | none. The row is a diagnostic and states no verdict | rejection 0.0530, 0.0364 to 0.0740 | reported |
| `refuter_reported` | `C-F__placebo_stale_rows` | diagnostic | 400 clusters of five, with m = 400 clusters (a limit that does not bind): the rejection rate of the placebo rule, with 5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.9000, 0.7420 to 0.9778 | reported |
| `refuter_reported` | `C-F__placebo_stale_rows_weak` | diagnostic | 400 clusters of five, with m = 400 clusters (a limit that does not bind): the rejection rate of the placebo rule, with 2.5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.4000, 0.2287 to 0.5908 | reported |
| `refuter_reported` | `C-F__subset` | diagnostic | 400 clusters of five, with m = 400 clusters (a limit that does not bind): the rejection rate of the subset rho rule against the sampled baseline | none. The row is a diagnostic and states no verdict | rejection 0.0550, 0.0381 to 0.0763 | reported |
| `refuter_reported` | `C-F__subset_double_se` | diagnostic | 400 clusters of five, with m = 400 clusters (a limit that does not bind): the rejection rate of the subset rule, at the sampled baseline's SE doubled | none. The row is a diagnostic and states no verdict | rejection 0.9800, 0.8606 to 0.9999 | reported |
| `refuter_reported` | `C-F__subset_double_se_weak` | diagnostic | 400 clusters of five, with m = 400 clusters (a limit that does not bind): the rejection rate of the subset rule, at the sampled baseline's SE times 1.43 | none. The row is a diagnostic and states no verdict | rejection 0.5600, 0.3709 to 0.7374 | reported |
| `refuter_reported` | `C-F__subset_half_se` | diagnostic | 400 clusters of five, with m = 400 clusters (a limit that does not bind): the rejection rate of the subset rule, at the sampled baseline's SE halved | none. The row is a diagnostic and states no verdict | rejection 1, 0.8995 to 1 | reported |
| `refuter_reported` | `C-F__subset_half_se_weak` | diagnostic | 400 clusters of five, with m = 400 clusters (a limit that does not bind): the rejection rate of the subset rule, at the sampled baseline's SE times 0.75 | none. The row is a diagnostic and states no verdict | rejection 0.4200, 0.2455 to 0.6101 | reported |
| `refuter_reported` | `C-H__control_confounded` | diagnostic | 800 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the negative-control rule, on a control confounded by U at 0.45 | none. The row is a diagnostic and states no verdict | rejection 0.9800, 0.8606 to 0.9999 | reported |
| `refuter_reported` | `C-H__control_confounded_weak` | diagnostic | 800 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the negative-control rule, on a control confounded by U at 0.15 | none. The row is a diagnostic and states no verdict | rejection 0.3000, 0.1491 to 0.4900 | reported |
| `refuter_reported` | `C-H__negative_control` | diagnostic | 800 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the negative-control rule on an outcome the treatment does not move | none. The row is a diagnostic and states no verdict | rejection 0.0430, 0.0282 to 0.0623 | reported |
| `refuter_reported` | `C-H__placebo` | diagnostic | 800 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the placebo t rule against the exact null 0 | none. The row is a diagnostic and states no verdict | rejection 0.0440, 0.0290 to 0.0635 | reported |
| `refuter_reported` | `C-H__placebo_stale_rows` | diagnostic | 800 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the placebo rule, with 5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.8200, 0.6427 to 0.9340 | reported |
| `refuter_reported` | `C-H__placebo_stale_rows_weak` | diagnostic | 800 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the placebo rule, with 2.5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.4200, 0.2455 to 0.6101 | reported |
| `refuter_reported` | `C-H__subset` | diagnostic | 800 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the subset rho rule against the sampled baseline | none. The row is a diagnostic and states no verdict | rejection 0.0480, 0.0323 to 0.0682 | reported |
| `refuter_reported` | `C-H__subset_double_se` | diagnostic | 800 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the subset rule, at the sampled baseline's SE doubled | none. The row is a diagnostic and states no verdict | rejection 1, 0.8995 to 1 | reported |
| `refuter_reported` | `C-H__subset_double_se_weak` | diagnostic | 800 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the subset rule, at the sampled baseline's SE times 1.43 | none. The row is a diagnostic and states no verdict | rejection 0.5000, 0.3155 to 0.6845 | reported |
| `refuter_reported` | `C-H__subset_half_se` | diagnostic | 800 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the subset rule, at the sampled baseline's SE halved | none. The row is a diagnostic and states no verdict | rejection 1, 0.8995 to 1 | reported |
| `refuter_reported` | `C-H__subset_half_se_weak` | diagnostic | 800 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the subset rule, at the sampled baseline's SE times 0.75 | none. The row is a diagnostic and states no verdict | rejection 0.4400, 0.2626 to 0.6291 | reported |
| `refuter_reported` | `C-T__control_confounded` | diagnostic | 4,000 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the negative-control rule, on a control confounded by U at 0.45 | none. The row is a diagnostic and states no verdict | rejection 0.9400, 0.7973 to 0.9931 | reported |
| `refuter_reported` | `C-T__control_confounded_weak` | diagnostic | 4,000 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the negative-control rule, on a control confounded by U at 0.15 | none. The row is a diagnostic and states no verdict | rejection 0.3400, 0.1800 to 0.5312 | reported |
| `refuter_reported` | `C-T__negative_control` | diagnostic | 4,000 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the negative-control rule on an outcome the treatment does not move | none. The row is a diagnostic and states no verdict | rejection 0.0460, 0.0306 to 0.0658 | reported |
| `refuter_reported` | `C-T__placebo` | diagnostic | 4,000 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the placebo t rule against the exact null 0 | none. The row is a diagnostic and states no verdict | rejection 0.0430, 0.0282 to 0.0623 | reported |
| `refuter_reported` | `C-T__placebo_stale_rows` | diagnostic | 4,000 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the placebo rule, with 5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.9000, 0.7420 to 0.9778 | reported |
| `refuter_reported` | `C-T__placebo_stale_rows_weak` | diagnostic | 4,000 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the placebo rule, with 2.5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.4200, 0.2455 to 0.6101 | reported |
| `refuter_reported` | `C-T__subset` | diagnostic | 4,000 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the subset rho rule against the sampled baseline | none. The row is a diagnostic and states no verdict | rejection 0.0420, 0.0274 to 0.0611 | reported |
| `refuter_reported` | `C-T__subset_double_se` | diagnostic | 4,000 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the subset rule, at the sampled baseline's SE doubled | none. The row is a diagnostic and states no verdict | rejection 0.9600, 0.8275 to 0.9979 | reported |
| `refuter_reported` | `C-T__subset_double_se_weak` | diagnostic | 4,000 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the subset rule, at the sampled baseline's SE times 1.43 | none. The row is a diagnostic and states no verdict | rejection 0.5600, 0.3709 to 0.7374 | reported |
| `refuter_reported` | `C-T__subset_half_se` | diagnostic | 4,000 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the subset rule, at the sampled baseline's SE halved | none. The row is a diagnostic and states no verdict | rejection 0.9800, 0.8606 to 0.9999 | reported |
| `refuter_reported` | `C-T__subset_half_se_weak` | diagnostic | 4,000 clusters of five and a sample of m = 400 whole clusters: the rejection rate of the subset rule, at the sampled baseline's SE times 0.75 | none. The row is a diagnostic and states no verdict | rejection 0.4400, 0.2626 to 0.6291 | reported |
| `refuter_reported` | `I1-F__control_confounded` | diagnostic | independent rows, n = 1,000 and m = 1,000 (a limit that does not bind): the rejection rate of the negative-control rule, on a control confounded by U at 0.45 | none. The row is a diagnostic and states no verdict | rejection 0.8400, 0.6665 to 0.9461 | reported |
| `refuter_reported` | `I1-F__control_confounded_weak` | diagnostic | independent rows, n = 1,000 and m = 1,000 (a limit that does not bind): the rejection rate of the negative-control rule, on a control confounded by U at 0.15 | none. The row is a diagnostic and states no verdict | rejection 0.2800, 0.1342 to 0.4689 | reported |
| `refuter_reported` | `I1-F__negative_control` | diagnostic | independent rows, n = 1,000 and m = 1,000 (a limit that does not bind): the rejection rate of the negative-control rule on an outcome the treatment does not move | none. The row is a diagnostic and states no verdict | rejection 0.0390, 0.0250 to 0.0576 | reported |
| `refuter_reported` | `I1-F__placebo` | diagnostic | independent rows, n = 1,000 and m = 1,000 (a limit that does not bind): the rejection rate of the placebo t rule against the exact null 0 | none. The row is a diagnostic and states no verdict | rejection 0.0470, 0.0315 to 0.0670 | reported |
| `refuter_reported` | `I1-F__placebo_stale_rows` | diagnostic | independent rows, n = 1,000 and m = 1,000 (a limit that does not bind): the rejection rate of the placebo rule, with 5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.8200, 0.6427 to 0.9340 | reported |
| `refuter_reported` | `I1-F__placebo_stale_rows_weak` | diagnostic | independent rows, n = 1,000 and m = 1,000 (a limit that does not bind): the rejection rate of the placebo rule, with 2.5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.2600, 0.1197 to 0.4474 | reported |
| `refuter_reported` | `I1-F__subset` | diagnostic | independent rows, n = 1,000 and m = 1,000 (a limit that does not bind): the rejection rate of the subset rho rule against the sampled baseline | none. The row is a diagnostic and states no verdict | rejection 0.0460, 0.0306 to 0.0658 | reported |
| `refuter_reported` | `I1-F__subset_double_se` | diagnostic | independent rows, n = 1,000 and m = 1,000 (a limit that does not bind): the rejection rate of the subset rule, at the sampled baseline's SE doubled | none. The row is a diagnostic and states no verdict | rejection 1, 0.8995 to 1 | reported |
| `refuter_reported` | `I1-F__subset_double_se_weak` | diagnostic | independent rows, n = 1,000 and m = 1,000 (a limit that does not bind): the rejection rate of the subset rule, at the sampled baseline's SE times 1.43 | none. The row is a diagnostic and states no verdict | rejection 0.4400, 0.2626 to 0.6291 | reported |
| `refuter_reported` | `I1-F__subset_half_se` | diagnostic | independent rows, n = 1,000 and m = 1,000 (a limit that does not bind): the rejection rate of the subset rule, at the sampled baseline's SE halved | none. The row is a diagnostic and states no verdict | rejection 1, 0.8995 to 1 | reported |
| `refuter_reported` | `I1-F__subset_half_se_weak` | diagnostic | independent rows, n = 1,000 and m = 1,000 (a limit that does not bind): the rejection rate of the subset rule, at the sampled baseline's SE times 0.75 | none. The row is a diagnostic and states no verdict | rejection 0.5400, 0.3522 to 0.7201 | reported |
| `refuter_reported` | `I1-T__control_confounded` | diagnostic | independent rows, n = 10,000 and a sample of m = 1,000: the rejection rate of the negative-control rule, on a control confounded by U at 0.45 | none. The row is a diagnostic and states no verdict | rejection 0.8400, 0.6665 to 0.9461 | reported |
| `refuter_reported` | `I1-T__control_confounded_weak` | diagnostic | independent rows, n = 10,000 and a sample of m = 1,000: the rejection rate of the negative-control rule, on a control confounded by U at 0.15 | none. The row is a diagnostic and states no verdict | rejection 0.1800, 0.0660 to 0.3573 | reported |
| `refuter_reported` | `I1-T__negative_control` | diagnostic | independent rows, n = 10,000 and a sample of m = 1,000: the rejection rate of the negative-control rule on an outcome the treatment does not move | none. The row is a diagnostic and states no verdict | rejection 0.0300, 0.0179 to 0.0468 | reported |
| `refuter_reported` | `I1-T__placebo` | diagnostic | independent rows, n = 10,000 and a sample of m = 1,000: the rejection rate of the placebo t rule against the exact null 0 | none. The row is a diagnostic and states no verdict | rejection 0.0520, 0.0356 to 0.0728 | reported |
| `refuter_reported` | `I1-T__placebo_stale_rows` | diagnostic | independent rows, n = 10,000 and a sample of m = 1,000: the rejection rate of the placebo rule, with 5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.8800, 0.7160 to 0.9681 | reported |
| `refuter_reported` | `I1-T__placebo_stale_rows_weak` | diagnostic | independent rows, n = 10,000 and a sample of m = 1,000: the rejection rate of the placebo rule, with 2.5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.2800, 0.1342 to 0.4689 | reported |
| `refuter_reported` | `I1-T__subset` | diagnostic | independent rows, n = 10,000 and a sample of m = 1,000: the rejection rate of the subset rho rule against the sampled baseline | none. The row is a diagnostic and states no verdict | rejection 0.0540, 0.0373 to 0.0751 | reported |
| `refuter_reported` | `I1-T__subset_double_se` | diagnostic | independent rows, n = 10,000 and a sample of m = 1,000: the rejection rate of the subset rule, at the sampled baseline's SE doubled | none. The row is a diagnostic and states no verdict | rejection 0.9800, 0.8606 to 0.9999 | reported |
| `refuter_reported` | `I1-T__subset_double_se_weak` | diagnostic | independent rows, n = 10,000 and a sample of m = 1,000: the rejection rate of the subset rule, at the sampled baseline's SE times 1.43 | none. The row is a diagnostic and states no verdict | rejection 0.5000, 0.3155 to 0.6845 | reported |
| `refuter_reported` | `I1-T__subset_half_se` | diagnostic | independent rows, n = 10,000 and a sample of m = 1,000: the rejection rate of the subset rule, at the sampled baseline's SE halved | none. The row is a diagnostic and states no verdict | rejection 0.9800, 0.8606 to 0.9999 | reported |
| `refuter_reported` | `I1-T__subset_half_se_weak` | diagnostic | independent rows, n = 10,000 and a sample of m = 1,000: the rejection rate of the subset rule, at the sampled baseline's SE times 0.75 | none. The row is a diagnostic and states no verdict | rejection 0.5000, 0.3155 to 0.6845 | reported |
| `refuter_reported` | `I2-F__control_confounded_weak` | diagnostic | independent rows, n = 2,000 and m = 2,000 (a limit that does not bind): the rejection rate of the negative-control rule, on a control confounded by U at 0.15 | none. The row is a diagnostic and states no verdict | rejection 0.6000, 0.4092 to 0.7713 | reported |
| `refuter_reported` | `I2-F__placebo_stale_rows_weak` | diagnostic | independent rows, n = 2,000 and m = 2,000 (a limit that does not bind): the rejection rate of the placebo rule, with 2.5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.5800, 0.3899 to 0.7545 | reported |
| `refuter_reported` | `I2-F__subset_double_se_weak` | diagnostic | independent rows, n = 2,000 and m = 2,000 (a limit that does not bind): the rejection rate of the subset rule, at the sampled baseline's SE times 1.43 | none. The row is a diagnostic and states no verdict | rejection 0.4800, 0.2976 to 0.6663 | reported |
| `refuter_reported` | `I2-F__subset_half_se_weak` | diagnostic | independent rows, n = 2,000 and m = 2,000 (a limit that does not bind): the rejection rate of the subset rule, at the sampled baseline's SE times 0.75 | none. The row is a diagnostic and states no verdict | rejection 0.5200, 0.3337 to 0.7024 | reported |
| `refuter_reported` | `I2-H__control_confounded_weak` | diagnostic | independent rows, n = 4,000 and a sample of m = 2,000: the rejection rate of the negative-control rule, on a control confounded by U at 0.15 | none. The row is a diagnostic and states no verdict | rejection 0.5000, 0.3155 to 0.6845 | reported |
| `refuter_reported` | `I2-H__placebo_stale_rows_weak` | diagnostic | independent rows, n = 4,000 and a sample of m = 2,000: the rejection rate of the placebo rule, with 2.5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.5800, 0.3899 to 0.7545 | reported |
| `refuter_reported` | `I2-H__subset_double_se_weak` | diagnostic | independent rows, n = 4,000 and a sample of m = 2,000: the rejection rate of the subset rule, at the sampled baseline's SE times 1.43 | none. The row is a diagnostic and states no verdict | rejection 0.4800, 0.2976 to 0.6663 | reported |
| `refuter_reported` | `I2-H__subset_half_se_weak` | diagnostic | independent rows, n = 4,000 and a sample of m = 2,000: the rejection rate of the subset rule, at the sampled baseline's SE times 0.75 | none. The row is a diagnostic and states no verdict | rejection 0.5000, 0.3155 to 0.6845 | reported |
| `refuter_reported` | `I2-T__control_confounded_weak` | diagnostic | independent rows, n = 20,000 and a sample of m = 2,000: the rejection rate of the negative-control rule, on a control confounded by U at 0.15 | none. The row is a diagnostic and states no verdict | rejection 0.5800, 0.3899 to 0.7545 | reported |
| `refuter_reported` | `I2-T__placebo_stale_rows_weak` | diagnostic | independent rows, n = 20,000 and a sample of m = 2,000: the rejection rate of the placebo rule, with 2.5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.5000, 0.3155 to 0.6845 | reported |
| `refuter_reported` | `I2-T__subset_double_se_weak` | diagnostic | independent rows, n = 20,000 and a sample of m = 2,000: the rejection rate of the subset rule, at the sampled baseline's SE times 1.43 | none. The row is a diagnostic and states no verdict | rejection 0.5400, 0.3522 to 0.7201 | reported |
| `refuter_reported` | `I2-T__subset_half_se_weak` | diagnostic | independent rows, n = 20,000 and a sample of m = 2,000: the rejection rate of the subset rule, at the sampled baseline's SE times 0.75 | none. The row is a diagnostic and states no verdict | rejection 0.4600, 0.2799 to 0.6478 | reported |
| `refuter_reported` | `P-20k__control_confounded` | diagnostic | independent rows, n = 20,000, refuted in full: the rejection rate of the negative-control rule, on a control confounded by U at 0.45 | none. The row is a diagnostic and states no verdict | rejection 1, 0.8995 to 1 | reported |
| `refuter_reported` | `P-20k__control_confounded_weak` | diagnostic | independent rows, n = 20,000, refuted in full: the rejection rate of the negative-control rule, on a control confounded by U at 0.15 | none. The row is a diagnostic and states no verdict | rejection 1, 0.8995 to 1 | reported |
| `refuter_reported` | `P-20k__negative_control` | diagnostic | independent rows, n = 20,000, refuted in full: the rejection rate of the negative-control rule on an outcome the treatment does not move | none. The row is a diagnostic and states no verdict | rejection 0.1000, 0.0222 to 0.2580 | reported |
| `refuter_reported` | `P-20k__placebo` | diagnostic | independent rows, n = 20,000, refuted in full: the rejection rate of the placebo t rule against the exact null 0 | none. The row is a diagnostic and states no verdict | rejection 0.0200, 0.000100 to 0.1394 | reported |
| `refuter_reported` | `P-20k__placebo_stale_rows` | diagnostic | independent rows, n = 20,000, refuted in full: the rejection rate of the placebo rule, with 5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 1, 0.8995 to 1 | reported |
| `refuter_reported` | `P-20k__placebo_stale_rows_weak` | diagnostic | independent rows, n = 20,000, refuted in full: the rejection rate of the placebo rule, with 2.5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 1, 0.8995 to 1 | reported |
| `refuter_reported` | `P-20k__subset` | diagnostic | independent rows, n = 20,000, refuted in full: the rejection rate of the subset rho rule against the sampled baseline | none. The row is a diagnostic and states no verdict | rejection 0.1000, 0.0222 to 0.2580 | reported |
| `refuter_reported` | `P-20k__subset_double_se` | diagnostic | independent rows, n = 20,000, refuted in full: the rejection rate of the subset rule, at the sampled baseline's SE doubled | none. The row is a diagnostic and states no verdict | rejection 1, 0.8995 to 1 | reported |
| `refuter_reported` | `P-20k__subset_double_se_weak` | diagnostic | independent rows, n = 20,000, refuted in full: the rejection rate of the subset rule, at the sampled baseline's SE times 1.43 | none. The row is a diagnostic and states no verdict | rejection 0.4200, 0.2455 to 0.6101 | reported |
| `refuter_reported` | `P-20k__subset_half_se` | diagnostic | independent rows, n = 20,000, refuted in full: the rejection rate of the subset rule, at the sampled baseline's SE halved | none. The row is a diagnostic and states no verdict | rejection 0.9400, 0.7973 to 0.9931 | reported |
| `refuter_reported` | `P-20k__subset_half_se_weak` | diagnostic | independent rows, n = 20,000, refuted in full: the rejection rate of the subset rule, at the sampled baseline's SE times 0.75 | none. The row is a diagnostic and states no verdict | rejection 0.5600, 0.3709 to 0.7374 | reported |
| `refuter_reported` | `P-4k__control_confounded` | diagnostic | independent rows, n = 4,000, refuted in full: the rejection rate of the negative-control rule, on a control confounded by U at 0.45 | none. The row is a diagnostic and states no verdict | rejection 1, 0.8995 to 1 | reported |
| `refuter_reported` | `P-4k__control_confounded_weak` | diagnostic | independent rows, n = 4,000, refuted in full: the rejection rate of the negative-control rule, on a control confounded by U at 0.15 | none. The row is a diagnostic and states no verdict | rejection 0.8200, 0.6427 to 0.9340 | reported |
| `refuter_reported` | `P-4k__negative_control` | diagnostic | independent rows, n = 4,000, refuted in full: the rejection rate of the negative-control rule on an outcome the treatment does not move | none. The row is a diagnostic and states no verdict | rejection 0.0200, 0.000100 to 0.1394 | reported |
| `refuter_reported` | `P-4k__placebo` | diagnostic | independent rows, n = 4,000, refuted in full: the rejection rate of the placebo t rule against the exact null 0 | none. The row is a diagnostic and states no verdict | rejection 0.0600, 0.0069 to 0.2027 | reported |
| `refuter_reported` | `P-4k__placebo_stale_rows` | diagnostic | independent rows, n = 4,000, refuted in full: the rejection rate of the placebo rule, with 5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 1, 0.8995 to 1 | reported |
| `refuter_reported` | `P-4k__placebo_stale_rows_weak` | diagnostic | independent rows, n = 4,000, refuted in full: the rejection rate of the placebo rule, with 2.5% of rows keeping their treatment | none. The row is a diagnostic and states no verdict | rejection 0.8200, 0.6427 to 0.9340 | reported |
| `refuter_reported` | `P-4k__subset` | diagnostic | independent rows, n = 4,000, refuted in full: the rejection rate of the subset rho rule against the sampled baseline | none. The row is a diagnostic and states no verdict | rejection 0.0800, 0.0138 to 0.2311 | reported |
| `refuter_reported` | `P-4k__subset_double_se` | diagnostic | independent rows, n = 4,000, refuted in full: the rejection rate of the subset rule, at the sampled baseline's SE doubled | none. The row is a diagnostic and states no verdict | rejection 1, 0.8995 to 1 | reported |
| `refuter_reported` | `P-4k__subset_double_se_weak` | diagnostic | independent rows, n = 4,000, refuted in full: the rejection rate of the subset rule, at the sampled baseline's SE times 1.43 | none. The row is a diagnostic and states no verdict | rejection 0.4800, 0.2976 to 0.6663 | reported |
| `refuter_reported` | `P-4k__subset_half_se` | diagnostic | independent rows, n = 4,000, refuted in full: the rejection rate of the subset rule, at the sampled baseline's SE halved | none. The row is a diagnostic and states no verdict | rejection 0.9600, 0.8275 to 0.9979 | reported |
| `refuter_reported` | `P-4k__subset_half_se_weak` | diagnostic | independent rows, n = 4,000, refuted in full: the rejection rate of the subset rule, at the sampled baseline's SE times 0.75 | none. The row is a diagnostic and states no verdict | rejection 0.5200, 0.3337 to 0.7024 | reported |
| `refuter_size` | `I2-F__negative_control` | positive | independent rows, n = 2,000 and m = 2,000 (a limit that does not bind): the negative-control rule on an outcome the treatment does not move | no more than 73 rejections in 1,000 trials | rejections 45 of 1000 | pass |
| `refuter_size` | `I2-F__placebo` | positive | independent rows, n = 2,000 and m = 2,000 (a limit that does not bind): the placebo t rule against the exact null 0 | no more than 73 rejections in 1,000 trials | rejections 50 of 1000 | pass |
| `refuter_size` | `I2-F__subset` | positive | independent rows, n = 2,000 and m = 2,000 (a limit that does not bind): the subset rho rule against the sampled baseline | no more than 73 rejections in 1,000 trials | rejections 54 of 1000 | pass |
| `refuter_size` | `I2-H__negative_control` | positive | independent rows, n = 4,000 and a sample of m = 2,000: the negative-control rule on an outcome the treatment does not move | no more than 73 rejections in 1,000 trials | rejections 58 of 1000 | pass |
| `refuter_size` | `I2-H__placebo` | positive | independent rows, n = 4,000 and a sample of m = 2,000: the placebo t rule against the exact null 0 | no more than 73 rejections in 1,000 trials | rejections 50 of 1000 | pass |
| `refuter_size` | `I2-H__subset` | positive | independent rows, n = 4,000 and a sample of m = 2,000: the subset rho rule against the sampled baseline | no more than 73 rejections in 1,000 trials | rejections 50 of 1000 | pass |
| `refuter_size` | `I2-T__negative_control` | positive | independent rows, n = 20,000 and a sample of m = 2,000: the negative-control rule on an outcome the treatment does not move | no more than 73 rejections in 1,000 trials | rejections 44 of 1000 | pass |
| `refuter_size` | `I2-T__placebo` | positive | independent rows, n = 20,000 and a sample of m = 2,000: the placebo t rule against the exact null 0 | no more than 73 rejections in 1,000 trials | rejections 46 of 1000 | pass |
| `refuter_size` | `I2-T__subset` | positive | independent rows, n = 20,000 and a sample of m = 2,000: the subset rho rule against the sampled baseline | no more than 73 rejections in 1,000 trials | rejections 46 of 1000 | pass |
| `refuter_size_relative` | `C-H__negative_control` | positive | 800 clusters of five and a sample of m = 400 whole clusters: the negative-control rule on an outcome the treatment does not move, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 43 of 1000, difference -0.0062 to 0.0222 | pass |
| `refuter_size_relative` | `C-H__negative_control__partner` | diagnostic | the full-data partner of 800 clusters of five and a sample of m = 400 whole clusters: the negative-control rule on an outcome the treatment does not move | none. The row is a diagnostic and states no verdict | rejections 35 of 1000, difference -0.0062 to 0.0222 | reported |
| `refuter_size_relative` | `C-H__placebo` | positive | 800 clusters of five and a sample of m = 400 whole clusters: the placebo t rule against the exact null 0, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 44 of 1000, difference -0.0248 to 0.0068 | pass |
| `refuter_size_relative` | `C-H__placebo__partner` | diagnostic | the full-data partner of 800 clusters of five and a sample of m = 400 whole clusters: the placebo t rule against the exact null 0 | none. The row is a diagnostic and states no verdict | rejections 53 of 1000, difference -0.0248 to 0.0068 | reported |
| `refuter_size_relative` | `C-H__subset` | positive | 800 clusters of five and a sample of m = 400 whole clusters: the subset rho rule against the sampled baseline, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 48 of 1000, difference -0.0233 to 0.0093 | pass |
| `refuter_size_relative` | `C-H__subset__partner` | diagnostic | the full-data partner of 800 clusters of five and a sample of m = 400 whole clusters: the subset rho rule against the sampled baseline | none. The row is a diagnostic and states no verdict | rejections 55 of 1000, difference -0.0233 to 0.0093 | reported |
| `refuter_size_relative` | `C-T__negative_control` | positive | 4,000 clusters of five and a sample of m = 400 whole clusters: the negative-control rule on an outcome the treatment does not move, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 46 of 1000, difference -0.0035 to 0.0255 | pass |
| `refuter_size_relative` | `C-T__negative_control__partner` | diagnostic | the full-data partner of 4,000 clusters of five and a sample of m = 400 whole clusters: the negative-control rule on an outcome the treatment does not move | none. The row is a diagnostic and states no verdict | rejections 35 of 1000, difference -0.0035 to 0.0255 | reported |
| `refuter_size_relative` | `C-T__placebo` | positive | 4,000 clusters of five and a sample of m = 400 whole clusters: the placebo t rule against the exact null 0, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 43 of 1000, difference -0.0257 to 0.0057 | pass |
| `refuter_size_relative` | `C-T__placebo__partner` | diagnostic | the full-data partner of 4,000 clusters of five and a sample of m = 400 whole clusters: the placebo t rule against the exact null 0 | none. The row is a diagnostic and states no verdict | rejections 53 of 1000, difference -0.0257 to 0.0057 | reported |
| `refuter_size_relative` | `C-T__subset` | positive | 4,000 clusters of five and a sample of m = 400 whole clusters: the subset rho rule against the sampled baseline, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 42 of 1000, difference -0.0288 to 0.0028 | pass |
| `refuter_size_relative` | `C-T__subset__partner` | diagnostic | the full-data partner of 4,000 clusters of five and a sample of m = 400 whole clusters: the subset rho rule against the sampled baseline | none. The row is a diagnostic and states no verdict | rejections 55 of 1000, difference -0.0288 to 0.0028 | reported |
| `refuter_size_relative` | `I1-T__negative_control` | positive | independent rows, n = 10,000 and a sample of m = 1,000: the negative-control rule on an outcome the treatment does not move, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 30 of 1000, difference -0.0224 to 0.0044 | pass |
| `refuter_size_relative` | `I1-T__negative_control__partner` | diagnostic | the full-data partner of independent rows, n = 10,000 and a sample of m = 1,000: the negative-control rule on an outcome the treatment does not move | none. The row is a diagnostic and states no verdict | rejections 39 of 1000, difference -0.0224 to 0.0044 | reported |
| `refuter_size_relative` | `I1-T__placebo` | positive | independent rows, n = 10,000 and a sample of m = 1,000: the placebo t rule against the exact null 0, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 52 of 1000, difference -0.0110 to 0.0210 | pass |
| `refuter_size_relative` | `I1-T__placebo__partner` | diagnostic | the full-data partner of independent rows, n = 10,000 and a sample of m = 1,000: the placebo t rule against the exact null 0 | none. The row is a diagnostic and states no verdict | rejections 47 of 1000, difference -0.0110 to 0.0210 | reported |
| `refuter_size_relative` | `I1-T__subset` | positive | independent rows, n = 10,000 and a sample of m = 1,000: the subset rho rule against the sampled baseline, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 54 of 1000, difference -0.0080 to 0.0240 | pass |
| `refuter_size_relative` | `I1-T__subset__partner` | diagnostic | the full-data partner of independent rows, n = 10,000 and a sample of m = 1,000: the subset rho rule against the sampled baseline | none. The row is a diagnostic and states no verdict | rejections 46 of 1000, difference -0.0080 to 0.0240 | reported |
| `refuter_size_relative` | `I2-H__negative_control` | positive | independent rows, n = 4,000 and a sample of m = 2,000: the negative-control rule on an outcome the treatment does not move, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 58 of 1000, difference -0.0033 to 0.0293 | pass |
| `refuter_size_relative` | `I2-H__negative_control__partner` | diagnostic | the full-data partner of independent rows, n = 4,000 and a sample of m = 2,000: the negative-control rule on an outcome the treatment does not move | none. The row is a diagnostic and states no verdict | rejections 45 of 1000, difference -0.0033 to 0.0293 | reported |
| `refuter_size_relative` | `I2-H__placebo` | positive | independent rows, n = 4,000 and a sample of m = 2,000: the placebo t rule against the exact null 0, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 50 of 1000, difference -0.0160 to 0.0160 | pass |
| `refuter_size_relative` | `I2-H__placebo__partner` | diagnostic | the full-data partner of independent rows, n = 4,000 and a sample of m = 2,000: the placebo t rule against the exact null 0 | none. The row is a diagnostic and states no verdict | rejections 50 of 1000, difference -0.0160 to 0.0160 | reported |
| `refuter_size_relative` | `I2-H__subset` | positive | independent rows, n = 4,000 and a sample of m = 2,000: the subset rho rule against the sampled baseline, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 50 of 1000, difference -0.0203 to 0.0123 | pass |
| `refuter_size_relative` | `I2-H__subset__partner` | diagnostic | the full-data partner of independent rows, n = 4,000 and a sample of m = 2,000: the subset rho rule against the sampled baseline | none. The row is a diagnostic and states no verdict | rejections 54 of 1000, difference -0.0203 to 0.0123 | reported |
| `refuter_size_relative` | `I2-T__negative_control` | positive | independent rows, n = 20,000 and a sample of m = 2,000: the negative-control rule on an outcome the treatment does not move, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 44 of 1000, difference -0.0162 to 0.0142 | pass |
| `refuter_size_relative` | `I2-T__negative_control__partner` | diagnostic | the full-data partner of independent rows, n = 20,000 and a sample of m = 2,000: the negative-control rule on an outcome the treatment does not move | none. The row is a diagnostic and states no verdict | rejections 45 of 1000, difference -0.0162 to 0.0142 | reported |
| `refuter_size_relative` | `I2-T__placebo` | positive | independent rows, n = 20,000 and a sample of m = 2,000: the placebo t rule against the exact null 0, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 46 of 1000, difference -0.0197 to 0.0117 | pass |
| `refuter_size_relative` | `I2-T__placebo__partner` | diagnostic | the full-data partner of independent rows, n = 20,000 and a sample of m = 2,000: the placebo t rule against the exact null 0 | none. The row is a diagnostic and states no verdict | rejections 50 of 1000, difference -0.0197 to 0.0117 | reported |
| `refuter_size_relative` | `I2-T__subset` | positive | independent rows, n = 20,000 and a sample of m = 2,000: the subset rho rule against the sampled baseline, beside its full-data partner | the 90% interval of the sampled rate minus the full-data rate lies inside -0.05 to 0.05 | rejections 46 of 1000, difference -0.0240 to 0.0080 | pass |
| `refuter_size_relative` | `I2-T__subset__partner` | diagnostic | the full-data partner of independent rows, n = 20,000 and a sample of m = 2,000: the subset rho rule against the sampled baseline | none. The row is a diagnostic and states no verdict | rejections 54 of 1000, difference -0.0240 to 0.0080 | reported |
<!-- /generated -->

## Result

All 39 gated readings pass. The absolute size of the I2 cells ranges from 0.044 to 0.058, inside
the limit of 73 of 1,000. Every one of the 15 relative intervals lies inside $\pm 0.05$; the widest
reaches 0.029. The strong defects are detected in 49 or 50 of 50 trials in each I2 cell. The
three mutation readings reject in 79% to 88% of null trials, so the study sees the sliced-fit
reference it is built to detect.

Sampling costs power exactly as refitting at $n = m$ would. On the weak defects the sampled I2-T
cell, at $m = 2000$ from $n = 20000$, detects the stale rows in 25 of 50 trials and the confounded
control in 29. The full fit at $n = 20000$ (P-20k) detects both in 50 of 50, and the full fit at
$n = 2000$ (I2-F) in 29 and 30. The SE rescaling defects read the same at every $n$, because the
subset `rho` is free of $n$.

Every I2 trial printed a verdict for each rule, and no I1 or clustered trial printed one.
`tests/unit/test_refuter_sampled_registry.py` recomputes every gated verdict and that scope from
the committed trials.

The primary truth row is red. Its bias interval, 0.0002 to 0.0149 in outcome units, crosses the
margin of 0.0100, and its SE-ratio interval, 1.066 to 1.373, crosses the screen limit of 1.20.
Coverage is 0.98. The study's own 1,000 I2-F trials fit the same configuration on other seeds:
they read a bias of 0.0027 with a Monte Carlo SE of 0.0015, an SE ratio of 1.015 and a coverage
of 0.950, and the point study's sanity row reads an SE ratio of 1.009. The 200-replication row is
therefore read as under-resolved, not as a defect of the fit, and no gated reading uses it. The
[red-cell ledger](red-cells.md) names the owner.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 200 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 1 | primary tests against truth |
| `independent_tests_passed` | 0 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 39 | repeated-sampling property cells |
| `property_cells_passed` | 39 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.1878 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9800 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 1.0658 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.3730 | highest bootstrap primary SE-ratio endpoint |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the level of the rule |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:type_i_ceiling` | 0.1000 | the rate a type-I cell must bound. No cell of this study reads it; a size cell reads its count |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power family must clear. No cell of this study reads it; a power cell reads its count |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 2 | the same bound, in primary replications |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The row publishes under the reporting policy, not gated | The fast tier recomputes each verdict and does not fail on a red one |
| There is no cross-implementation evidence | No maintained implementation refutes a sample of a fit's units |
| The claim needs independent units | The exactness argument needs i.i.d. subjects, or i.i.d. clusters declared with `id=`. Rows that depend on each other without a declared cluster are outside it |
| One law and one fit | The study reads the T1 configuration only. Other estimators rest on the exactness argument and on the verdict rows at their own sampled configuration |
| Gross miscalibration only | A size reading of 1,000 trials does not certify the size to two decimals. A rule of true size 0.06 fails the absolute count with probability about 0.30 across nine readings |
| A sample loses power | A sampled verdict has the power of a fit at $n = m$, not at $n$. The weak defects show the gap |
| The primary phase is under-resolved | 200 replications cannot put the 99% bias and SE-ratio intervals inside their margins. The truth row is red with an owner, and no rule row reads it |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_sampled/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_sampled/manifest.json)
records the seeds, the margins, the laws, the source hashes and the result hashes. The
[trials](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_sampled/trials.csv)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_sampled/properties.csv)
carry every published row.
