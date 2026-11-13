# Ratio trimming on LTMLE and SDR

This study validates the point estimate of `LTMLE(ratio_trim=q)` and `SDR(ratio_trim=q)`, which
cap every per-node density ratio at the empirical quantile $q$ of all per-node ratios, zeros
included, and build the clever covariates from the capped ratios. The
[ratio trimming section](../longitudinal-tmle.md#ratio-trimming) states the option.

Trimming changes the estimator, not the estimand. The trimmed fit's error is
$\sum_t P_0[(\tilde R_t - R_t)(\bar Q^*_{t+1} - \bar Q^*_t)]$ plus second-order terms, where
$\tilde R_t$ is the trimmed and $R_t$ the true cumulative ratio. That term vanishes when every
iterated outcome regression is correct, and it stays on the trimmed share when one is wrong.
So the trimmed point estimate is consistent when the outcome regressions are, and the trim
removes double robustness on the trimmed share. A trimmed fit withholds its standard error
(`trimmed_ratio_plugin`, roadmap F30): the study reads the point estimate only.

No canonical implementation is compared. An `lmtp` pairing of the trimmed fits is possible later and
blocks nothing. The study rests on exact limits.

## What was fitted

Law LE draws $W \sim U(0, 1)$, $A_1 \sim \operatorname{Bernoulli}(\operatorname{expit}(-3 + 4W))$,
retention at each node with probability $\operatorname{expit}(2.5 + 0.5 W)$,
$L_2 \sim \operatorname{Bernoulli}(0.2 + 0.3 W + 0.2 A_1)$,
$A_2 \sim \operatorname{Bernoulli}(\operatorname{expit}(-2.5 + 3 W + 0.5 L_2 + 0.5 A_1))$ and
$Y \sim \operatorname{Bernoulli}(0.05 + 0.6 W + 0.15 L_2 + 0.05 A_1 + 0.1 A_2)$. Always-treat has
weak support at small $W$, so the cumulative ratio is heavy-tailed. The target is
$E[Y^{\text{always}}] = 0.5825$ at n = 2,000.

Every fit uses logistic treatment and censoring mechanisms by Newton, which are correct. The
correct outcome regression is `LinearRegression`, which contains every iterated regression of
LE. The wrong one is a linear regression that drops $W$. The gated cells trim at 0.95, where the
population threshold is 4.4506 and the population trimmed share 0.1578. All fits of a replicate
read one draw, so the trimmed and untrimmed fits, the two outcome regressions and the two
estimators differ on the same data in the setting alone.

The law, learners and exact limits are
in `tests/studies/_option_trim_law.py`. The exact limits are weighted fits on a quadrature form
of the law at the law's trim quantile, split at the kinks of the trimmed ratio, and
`tests/unit/test_option_ratio_trim_design.py` recomputes them.
`tests/unit/test_option_ratio_trim_witness.py` checks the remainder identity at $10^{-8}$, with
mutation controls that drop the trim from the clever covariate, carry it to node 2 only where
node 2 was trimmed, or take the quantile of the positive entries.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: in-sample LTMLE with ratio_trim=0.95, linear outcome regressions and logistic mechanisms; the point estimate only | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` | -0.0014 to 0.0044 | 0.7281 | 0.5743 | pass |
<!-- /generated -->

The primary row is point-only, so it passes on its bias alone. Its coverage and SE ratio come
from the plug-in diagnostic standard error, which is not an interval the package reports.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `trim_necessity` | `ltmle_trimmed` | control | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: LTMLE with ratio_trim=0.95 and an outcome regression that ignores W, in sample | the 99% interval of the standardized bias must lie wholly beyond the margin | bias 0.0673 to 0.0721, margin 0.0092 | pass |
| `trim_necessity` | `ltmle_untrimmed` | positive | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: LTMLE untrimmed with the same outcome regression that ignores W, in sample | bias interval inside the equivalence margin | bias -0.000836 to 0.0070, margin 0.0151 | pass |
| `trim_necessity` | `sdr_trimmed` | control | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: SDR with ratio_trim=0.95 and an outcome regression that ignores W, in sample | the 99% interval of the standardized bias must lie wholly beyond the margin | bias 0.0825 to 0.0870, margin 0.0087 | pass |
| `trim_necessity` | `sdr_untrimmed` | positive | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: SDR untrimmed with the same outcome regression that ignores W, in sample | bias interval inside the equivalence margin | bias -0.0032 to 0.0056, margin 0.0170 | pass |
| `trim_point` | `ltmle_in_sample` | positive | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: LTMLE with ratio_trim=0.95, linear outcome regressions (correct on this law), in sample | bias interval inside the equivalence margin | bias -0.0019 to 0.0040, margin 0.0115 | pass |
| `trim_point` | `ltmle_stacked` | positive | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: LTMLE with ratio_trim=0.95, linear outcome regressions (correct on this law), five-fold cross-fitted | bias interval inside the equivalence margin | bias -0.0018 to 0.0041, margin 0.0115 | pass |
| `trim_point` | `sdr_in_sample` | positive | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: SDR with ratio_trim=0.95, linear outcome regressions (correct on this law), in sample | bias interval inside the equivalence margin | bias -0.0019 to 0.0041, margin 0.0117 | pass |
| `trim_point` | `sdr_stacked` | positive | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: SDR with ratio_trim=0.95, linear outcome regressions (correct on this law), five-fold cross-fitted | bias interval inside the equivalence margin | bias -0.0019 to 0.0041, margin 0.0117 | pass |
| `trim_reported` | `ltmle_right_trim99` | diagnostic | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: LTMLE with ratio_trim=0.99, linear outcome regressions, in sample | none. The row is a diagnostic and states no verdict | bias -0.0022 to 0.0045, coverage 0.8788, SE ratio 0.8331, exact bias 0 | reported |
| `trim_reported` | `ltmle_right_untrimmed` | diagnostic | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: LTMLE untrimmed, linear outcome regressions, in sample | none. The row is a diagnostic and states no verdict | bias -0.0029 to 0.0044, coverage 0.8956, SE ratio 0.9285, exact bias 0 | reported |
| `trim_reported` | `ltmle_wrong_trim99` | diagnostic | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: LTMLE with ratio_trim=0.99, an outcome regression that ignores W, in sample | none. The row is a diagnostic and states no verdict | bias 0.0180 to 0.0246, coverage 0.8844, SE ratio 0.9106, exact bias 0.0184 | reported |
| `trim_reported` | `sdr_right_trim99` | diagnostic | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: SDR with ratio_trim=0.99, linear outcome regressions, in sample | none. The row is a diagnostic and states no verdict | bias -0.0022 to 0.0045, coverage 0.8800, SE ratio 0.8331, exact bias 0 | reported |
| `trim_reported` | `sdr_right_untrimmed` | diagnostic | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: SDR untrimmed, linear outcome regressions, in sample | none. The row is a diagnostic and states no verdict | bias -0.0029 to 0.0045, coverage 0.9050, SE ratio 0.9394, exact bias 0 | reported |
| `trim_reported` | `sdr_wrong_trim99` | diagnostic | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: SDR with ratio_trim=0.99, an outcome regression that ignores W, in sample | none. The row is a diagnostic and states no verdict | bias 0.0236 to 0.0305, coverage 0.8769, SE ratio 0.9809, exact bias 0.0250 | reported |
| `trim_reported` | `trimmed_share` | diagnostic | law LE (W uniform, two binary treatment nodes with weak support for always-treat at small W, censoring at both nodes), the mean under always, n = 2,000: the share of per-node ratios the 0.95 trim caps in the in-sample LTMLE fit, read against its population value | none. The row is a diagnostic and states no verdict | mean share 0.1579 against 0.1578 | reported |
<!-- /generated -->

The four `trim_point` cells are the claim: with correct outcome regressions the trimmed fits are
unbiased in sample and cross-fitted, for LTMLE and SDR. The `trim_necessity` cells are its
control. With the outcome regression that drops $W$, the trimmed fits miss by 1.90 and 2.43
SDs, against exact limits of 0.0679 and 0.0832 in outcome units, while the untrimmed fits with
the same regression stay unbiased, because the untrimmed ratio corrects the outcome error.

## Reported readings

These rows carry no verdict.

**The plug-in diagnostic.** The trimmed fits' plug-in standard error is 0.55 to 0.58 of the
spread of the trimmed estimate, and the nominal 95% interval built on it covers 0.72 to 0.73.
It is a diagnostic, not a standard error, which is why the interval is withheld.

**The 0.99 trim.** Trimmed at 0.99 with the outcome regression that drops $W$, the fits read
0.41 and 0.50 SD against exact limits of 0.0184 and 0.0250. A smaller trimmed share leaves a
smaller remainder, below what 1,600 replicates resolve as a control.

**The default untrimmed fit under-covers on this law at n = 2,000.** With correct outcome
regressions the untrimmed LTMLE covers 0.896 and the untrimmed SDR 0.905, with SE ratios 0.928
and 0.939 and no bias. The pilot read 0.894 and 0.903 on 4,000 draws of a different seed, so the
shortfall is confirmed on 5,600 disjoint draws. On this law the cumulative ratio is
heavy-tailed: about 0.2% of units carry most of the influence-curve variance, so a sample of
2,000 often holds too few of them and the influence-curve standard error understates the spread.
The reading turns no cell red, and it is not evidence about the trimmed fit.

No registered cell measures a larger n. A separate diagnosis outside this study, with the same
law, learners and settings on 2,000 draws per size, found the shortfall to be finite-sample.
Coverage rose to 0.946 for both estimators at n = 16,000, and $n$ times the mean squared
standard error reached 6.23 (LTMLE) and 6.26 (SDR) against the exact efficient variance 6.25.

## Sizing

Every cell and the primary phase run 1,600 replicates of one stream. At the pilot biases the six
gated positives pass jointly with probability at least 0.95, and the two controls with
probability above 0.9999. The pilot is `tests/studies/option_ratio_trim_pilot.py`, on a stream no
declared cell reads, and the design test recomputes the rule.

## Red-cell route

The study is registered under `reporting`, and no cell is red. The declared reading is in
`tests/canonical/option_ratio_trim/regenerate.py`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 1 | truth tests passing |
| `independent_tests_total` | 1 | truth tests reported |
| `property_cells_passed` | 8 | gated property cells passing |
| `property_cells_total` | 8 | gated property cells reported |
| `max_standardized_bias` | 0.0333 | largest primary standardized bias |
| `min_coverage` | 0.7281 | lowest primary coverage of the plug-in diagnostic |
| `properties[trim_necessity/ltmle_trimmed]:standardized_bias` | 1.8996 | trimmed LTMLE with the outcome regression that drops W |
| `properties[trim_necessity/sdr_trimmed]:standardized_bias` | 2.4281 | trimmed SDR with the outcome regression that drops W |
| `properties[trim_necessity/ltmle_trimmed]:exact_bias` | 0.0679 | exact limit of the trimmed LTMLE control |
| `properties[trim_point/sdr_in_sample]:se_ratio` | 0.5538 | lowest plug-in SE ratio of a trimmed fit |
| `properties[trim_point/ltmle_stacked]:se_ratio` | 0.5805 | highest plug-in SE ratio of a trimmed fit |
| `properties[trim_point/sdr_in_sample]:coverage` | 0.7156 | lowest plug-in coverage of a trimmed fit |
| `properties[trim_point/ltmle_stacked]:coverage` | 0.7312 | highest plug-in coverage of a trimmed fit |
| `properties[trim_reported/ltmle_right_untrimmed]:coverage` | 0.8956 | coverage of the untrimmed LTMLE with correct regressions |
| `properties[trim_reported/sdr_right_untrimmed]:coverage` | 0.9050 | coverage of the untrimmed SDR with correct regressions |
| `properties[trim_reported/ltmle_right_untrimmed]:se_ratio` | 0.9285 | SE ratio of the untrimmed LTMLE |
| `properties[trim_reported/sdr_right_untrimmed]:se_ratio` | 0.9394 | SE ratio of the untrimmed SDR |
| `properties[trim_reported/trimmed_share]:mean_estimate` | 0.1579 | mean trimmed share of the 0.95 trim |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor. No cell of this study reads it |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold. No cell of this study reads it |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen. No cell of this study reads it |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen. No cell of this study reads it |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound. No cell of this study reads it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound. No cell of this study reads it |
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
| `margin:subject_failures_allowed` | 16 | the same bound, in primary replications |

## Limitations

- **Consistent only with correct outcome regressions.** The trim removes double robustness on
  the trimmed share, so a wrong outcome regression leaves a bias that the untrimmed fit would
  correct.
- **Measured with a linear outcome regression that is correct.** The `G56` rows pin
  `LinearRegression` for the outcome and pseudo-outcome regressions, so another learner reads
  experimental. Pinning the class does not make a linear regression correct on other data.
- **No interval.** The trimmed fit withholds its standard error (F30), and its plug-in
  diagnostic understates the spread.
- **One two-node law with censoring and one regimen.** Other node counts, regimens, survival
  outcomes, the hazard path, weights and clusters with the option read experimental.
- **Quantiles 0.95 and 0.99 only.** The gated cells trim at 0.95; the 0.99 rows are reported.
- **No comparator.** No `lmtp` pairing has run.

## Reproduction

`tests/canonical/option_ratio_trim/regenerate.py` regenerates the study and states the run form.
It runs no reference process, because the study has no comparator. The manifest records the
seeds, the margins, the estimator configuration and the source and result hashes, and the run
log records the commit, the clean tree and the wall time.
