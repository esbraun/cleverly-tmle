# Method evidence studies

Each section here is one row of the [method evidence grid](evidence.md#method-evidence-grid).
A row exists only for a study registered in `tests/studies/evidence/registry.py`; the shared
machinery in `tests/studies/evidence/` computes every verdict and every number quoted below, and
`tests/unit/test_method_evidence.py` checks this document against the committed results rather
than trusting it. See [adding a method row](evidence.md#adding-a-method-row).

## Canonical point-treatment TMLE

This is the first method-level evidence row: ordinary, non-cross-fitted point-treatment TMLE in
cleverly compared with R [`tmle3`](https://github.com/tlverse/tmle3) 0.2.0 at pinned commit
[`ed72f8a`](https://github.com/tlverse/tmle3/tree/ed72f8a20e64c914ab25ffe015d865f7a9963d27).
The comparison is a bounded implementation witness. The independent claims come from the
parameter oracles and from repeated-sampling checks of the properties in van der Laan & Rubin's
[original TMLE paper](https://doi.org/10.2202/1557-4679.1043).

### What was compared

Python generates every realized dataset and gives the same rows and all DGP covariates to both
implementations. Both use ordinary TMLE, corresponding GLM nuisance regressions, pointwise 95%
intervals, treatment 1 versus 0, comfortable overlap, and no simultaneous adjustment. The
published run uses the replication count and sample size in the table below, for each of two
laws:

- a bounded continuous-outcome law with effect modification, covering `ey1`, `ey0`, `ate`, `att`,
  `atc`, `ey_obs`, and `par`;
- a binary-outcome law covering those parameters plus `paf`, `rr`, and `or`.

The [per-replication results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/replicates.csv.gz)
contain the pairing key, truth, estimate, standard error, native interval endpoints, interval scale,
and coverage indicator. The [performance summary](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/summary.csv),
the [independent performance tests](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/performance-tests.csv)
and the [paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/equivalence.csv)
are all recomputed from that file by the test suite. Testing each implementation against the law
separately is what stops agreement between two poor implementations from counting as evidence.

### How a verdict is reached

Every accept-decision is bounded by a margin declared before the run, and none of them is a test
of whether a discrepancy is exactly zero. That distinction is the design, not a detail. A Monte
Carlo study accumulates evidence by adding replications, so a verdict has to become easier -- or
at worst stay put -- as replications grow. A significance test does the reverse: it converges on
rejecting any estimator whose finite-sample remainder is not identically zero, which is every
estimator, so the study would eventually go red for the one reason that is not a defect. An
earlier version of this study carried two such rules, and quadrupling its replication count would
have failed it without a line of estimator code changing.
`tests/unit/test_evidence_framework.py` holds both rules side by side and asserts which way each
one moves.

Independently, per implementation and estimand, at 99% confidence:

- **bias** -- the Student interval for the error on the implementation's own reported scale must
  lie inside 0.25 empirical standard deviations of zero;
- **coverage** -- the lower endpoint of the exact Clopper--Pearson interval must clear 0.90. This
  is one-sided. Whether a nominal 95% interval is *valid* is the question; whether it is 95% to
  the third decimal is a different question that no finite study answers affirmatively;
- **reported standard error** -- the bootstrap interval for mean reported SE over empirical SD
  must lie inside a 0.80--1.20 sanity band. This band is deliberately wider than the coverage
  floor implies -- a ratio of 0.80 corresponds to about 88% coverage -- so the coverage gate binds
  first and the band screens for a standard error that is wrong by a different order of magnitude.
  The resolution the band actually achieved is published below, because a wide margin must not
  read as a tight calibration proof.

Neither of those two is a *calibration* claim, and the difference matters: a reported standard error
uniformly a tenth too small leaves true coverage at 0.922, which clears the 0.90 floor and sits well
inside the sanity band. Calibration is asked separately, of one property cell where both nuisance
regressions are correctly specified and the theory therefore does promise that the influence-curve
standard error is the efficient one. There the 99% resampling interval for the SE ratio must lie
inside 0.93--1.07 and the exact coverage interval inside 0.92--0.98, two-sided in both cases. That
is still an equivalence statement rather than a point test -- it asks whether the interval lies
inside a declared band, which more replications make easier, not whether coverage *is* 0.95, which
they would make impossible.

Across implementations, on the pairing, also at 99%: the paired mean difference must lie within
0.15 pooled empirical standard deviations, which is symmetric because a large difference in either
direction means the two are not computing the same thing. The remaining comparisons are one-sided
non-inferiority tests, because cleverly performing better than the reference is a result rather
than a failure: the bootstrap upper bound for its RMSE ratio must be at most 1.10, the lower bound
for its coverage difference at least -0.025, and the upper bound for excess absolute SE-calibration
error at most 0.05 where the SE scales are comparable. The published verdict is exactly those two
claims. Whether each implementation is any good on its own is carried as two further columns, so a
reference that degrades is reported against the reference instead of turning cleverly's row red.

The harness has negative controls for all of it. The fast tests independently corrupt bias,
coverage and reported standard errors for each implementation in turn, require only that
implementation to fail, and require the untouched one to keep passing; a paired mutation makes
cleverly materially worse and must fail both similarity and non-inferiority; and a reference-only
mutation must fail the reference's column while leaving cleverly's standing.

### Measured values

Every figure this section quotes is resolved by name and checked at the precision it is printed to,
so a stale or mistyped number is a test failure rather than a reading error. Names beginning
`margin:` are the thresholds declared *before* the run; everything else is measured from the
committed results. Both resolve the same way, so moving a declared threshold in the code changes
this table rather than leaving it asserting a rule the study never applied.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | replications per law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 34 | implementation-estimand tests against truth |
| `independent_tests_passed` | 34 | of those, passing |
| `paired_tests_total` | 17 | scenario-estimand cells compared with `tmle3` |
| `paired_tests_passed` | 17 | of those, passing |
| `property_cells_total` | 12 | repeated-sampling property cells |
| `property_cells_passed` | 12 | of those, passing |
| `max_standardized_bias` | 0.1254 | largest bias, in empirical standard deviations |
| `min_coverage` | 0.9344 | lowest measured coverage of a nominal 95% interval |
| `min_coverage_ci_lower` | 0.9168 | lowest exact 99% coverage endpoint, against a floor of 0.90 |
| `min_se_ratio_ci_lower` | 0.9040 | lowest bootstrap SE-ratio endpoint, against a band of 0.80--1.20 |
| `max_se_ratio_ci_upper` | 1.0649 | highest bootstrap SE-ratio endpoint |
| `max_se_ratio_resolution` | 0.0960 | how far from 1.0 the widest SE-ratio interval still reaches |
| `max_margin_utilization` | 0.7445 | largest share of the paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0088 | largest one-sided RMSE-ratio bound, against a margin of 1.10 |
| `min_coverage_difference_lower` | -0.0081 | smallest one-sided coverage-difference bound, against -0.025 |
| `max_calibration_excess_upper` | 0.0129 | largest comparable SE-calibration-excess bound, against 0.05 |
| `summary_cells` | 34 | summary cells over both laws and both implementations |
| `cells_with_se_ratio_below_one` | 13 | of those, reporting a standard error below the empirical spread |
| `cells_with_coverage_below_nominal` | 17 | of those, covering below 95% |
| `summary[cleverly/continuous/par]:coverage` | 0.9344 | the study's lowest coverage, cleverly |
| `summary[tmle3/continuous/par]:coverage` | 0.9375 | the same cell in `tmle3` |
| `summary[cleverly/continuous/att]:bias` | -0.000144 | continuous ATT bias against known truth, cleverly |
| `summary[tmle3/continuous/att]:bias` | -0.000888 | the same, `tmle3` |
| `summary[cleverly/continuous/atc]:bias` | 0.000054 | continuous ATC bias against known truth, cleverly |
| `summary[tmle3/continuous/atc]:bias` | 0.000837 | the same, `tmle3` |
| `continuous_summary_cells` | 14 | continuous-law summary cells, both implementations |
| `continuous_cells_with_se_ratio_below_one` | 8 | of those, reporting a standard error below the empirical spread |
| `continuous_cells_with_coverage_below_nominal` | 7 | of those, covering below 95% |
| `properties[double_robustness/both_correct]:bias` | -0.0034 | bias with both nuisances correct |
| `properties[double_robustness/outcome_correct]:bias` | 0.000306 | bias with only the outcome regression correct |
| `properties[double_robustness/treatment_correct]:bias` | -0.0182 | bias with only the treatment mechanism correct |
| `properties[double_robustness/both_wrong]:bias` | -0.3332 | bias with both nuisances wrong -- the negative control |
| `properties[double_robustness/both_wrong]:coverage` | 0.1967 | coverage of the same control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5045 | fitted log-log contraction rate of the sampling spread |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5374 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4725 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.4986 | the same rate for the mean reported standard error |
| `properties[root_n_and_efficiency/n_500]:se_ratio` | 0.9913 | SE calibration at n = 500 |
| `properties[root_n_and_efficiency/n_2000]:se_ratio` | 0.9777 | SE calibration at n = 2,000 |
| `properties[root_n_and_efficiency/n_8000]:se_ratio` | 1.0077 | SE calibration at n = 8,000 |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 1.0027 | SE calibration where both nuisances are correct |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9679 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0385 | its 99% upper endpoint |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9542 | coverage of the same cell |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0325 | rejection under a confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0627 | its 99% upper endpoint, against 0.05 + 0.05 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under a real effect -- the positive control |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval below |
| `margin:alpha` | 0.0500 | nominal size of the estimator's own intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the exact coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size the one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest RMSE ratio the one-sided upper bound may reach |
| `margin:coverage_noninferiority` | -0.0250 | smallest coverage difference the one-sided lower bound may reach |
| `margin:calibration_noninferiority` | 0.0500 | largest excess SE-calibration error the upper bound may reach |
| `margin:minimum_power` | 0.8000 | rejection lower bound the power control must clear |
| `margin:root_n_slope` | -0.5000 | the contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | the slower rate the interval must exclude |

### What the study establishes, and what it does not

The counts above are the claim: every implementation-estimand test against truth passes, every
paired similarity and non-inferiority test passes, and every property cell passes. Within the
declared practical margins cleverly is established to be as good as the pinned R reference, and it
is not penalized where it is better.

Three things the run measured are worth stating plainly rather than leaving in a CSV.

**Interval calibration shows no systematic direction.** The reported standard error falls below the
empirical spread of the estimates in `cells_with_se_ratio_below_one` of `summary_cells` cells and
above it in the rest, and coverage falls below nominal in `cells_with_coverage_below_nominal` of
them -- about half each way, which is what Monte Carlo scatter looks like. An earlier run of this
study at a quarter of the replication count had every continuous-law cell below one and was read as
a systematic understatement of uncertainty. It was not: raising the replication count dissolved it.
That is the reason this section quotes counts computed from the results rather than impressions
formed by looking at them.

**Where coverage does sit low, both implementations sit low together.** The lowest in the study is
continuous-law `par`, at `summary[cleverly/continuous/par]:coverage` for cleverly against
`summary[tmle3/continuous/par]:coverage` for `tmle3` -- clear of the declared floor, and equal
between the two to within Monte Carlo error. Whatever remains there is a property of ordinary
non-cross-fitted TMLE on this law rather than of either implementation. The
`root_n_and_efficiency` cells are the discriminator: SE calibration is measured at three sample
sizes, so a gap that shrinks with `n` is finite-sample nuisance noise while one that persisted
would be a wrong variance formula.

**The standard-error band is a screen, not a calibration proof.** The widest bootstrap SE-ratio
interval still reaches `max_se_ratio_resolution` from 1.0, so a systematic misstatement smaller
than about a tenth is not something these 34 cells can rule out. Coverage carries the validity claim
instead, where the exact interval is tight enough to separate the nominal rate from a two-point
shortfall. Calibration is ruled on where it can be: the `interval_calibration` cell puts the SE
ratio at `properties[interval_calibration/correctly_specified]:se_ratio` with a 99% interval from
`properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` to
`properties[interval_calibration/correctly_specified]:se_ratio_ci_upper`, which excludes the
tenth-scale misstatement the screen above cannot.

Continuous-law ATT and ATC use the most of the paired similarity margin -- `max_margin_utilization`
of it at the widest -- and stay inside it. That difference was evaluated as a possible cleverly
defect and does not justify a production fix: measured against the known truth, cleverly's absolute
bias on those two estimands is an order of magnitude smaller than the constrained R path's
(`summary[cleverly/continuous/att]:bias` against `summary[tmle3/continuous/att]:bias` for ATT, and
`summary[cleverly/continuous/atc]:bias` against `summary[tmle3/continuous/atc]:bias` for ATC), with
essentially unchanged RMSE and coverage. The gate stays asymmetric so that a future change
reversing this conclusion fails the suite rather than being documented away.

### Confidence intervals by estimand

| estimands | cleverly interval | R `tmle3` interval | comparison |
| --- | --- | --- | --- |
| levels, ATE, ATT, ATC, PAR | identity-scale Wald | identity-scale Wald | estimate, SE, endpoints, and coverage |
| RR and OR | log-scale Wald, exponentiated | log-scale Wald, exponentiated | estimate, SE, endpoints, and coverage |
| PAF | identity-scale Wald from the PAF influence curve | negative-log-complement (log-risk-ratio) Wald transformed by `1 - exp(-x)` | point performance and coverage; raw SE parity is not claimed |

That PAF distinction is not papered over. The two intervals have the same first-order delta-method
limit but need not have the same finite-sample endpoints, so forcing their raw standard errors into
one parity threshold would compare different reported scales. The exemption is declared on the
study record and a test requires the two implementations to genuinely report different scales for
it, so an exemption cannot be claimed where it is not earned.

### Properties checked independently

The [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/properties.csv)
and their [replication-level data](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/property-replicates.csv.gz)
record four nuisance-model cells, three sample sizes, a fitted rate, a correctly-specified
calibration cell, a sharp-null experiment and a power control:

| claim | positive evidence | negative control or discriminator |
| --- | --- | --- |
| double-robust consistency | with both nuisances correct, with only the outcome regression correct, and with only the treatment mechanism correct, the 99% interval for the bias lies inside 0.25 empirical standard deviations of zero | with both nuisances wrong the same interval must lie entirely *outside* that margin -- the same instrument in both directions, so a study too small to say anything fails both halves rather than passing the first |
| root-n behaviour | the log empirical spread of the estimates, regressed on log `n` across three sample sizes, has a 99% interval lying within 0.125 of -1/2 -- half the distance to the alternative below, so the accept and the reject verdicts partition the two rates the claim is about | that interval must also exclude -1/4, so a merely decreasing spread fails. The margin is what keeps this an equivalence statement: requiring the interval to *contain* -1/2 is a point test, and at these replication counts the reported-SE rate is resolved to about a thousandth and would fail it. That rate is fitted separately and labelled as what it is -- an influence-curve SE is sigma-hat over root n, so its rate is near arithmetic and catches only a standard error carrying the wrong power of `n` |
| local efficiency and inference | at each size, the SE ratio is inside the sanity band, coverage clears the floor, and the bias interval is inside the equivalence margin | SE calibration is checked separately from bias, so correct centering cannot hide the wrong width, and it is measured at three sizes so a finite-sample gap is distinguishable from a wrong formula |
| interval calibration | on a law where a GLM is correctly specified for both nuisances, the 99% resampling interval for the SE ratio lies inside 0.93--1.07 and the exact coverage interval inside 0.92--0.98 | both halves are required and neither implies the other: a standard error inflated by a constant keeps coverage inside its band while failing the ratio, and this is the only gate in the study that a uniform tenth-scale understatement fails -- the coverage floor is one-sided and the sanity band cannot be tightened past what that floor implies |
| type-I error | the 99% upper endpoint of the ATE rejection rate under a sharp null clears the nominal size by no more than the declared margin | the null law retains its confounding, so the test is not an unadjusted randomized comparison; and a separate cell under a real effect must reject, so a test that never fires cannot pass by being inert |

These statistical checks complement rather than replace the exact-law, Gateaux, remainder, and
identity instruments in the [estimand evidence manifest](evidence.md#the-table).

### Reproduction and provenance

The complete regeneration instructions and the PAF qualification are in the
[`tmle3` fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/README.md).
The container pins R 4.5.2 by digest, `tmle3` at `ed72f8a`, and `sl3` at
[`0e8f236`](https://github.com/tlverse/sl3/tree/0e8f2365bcbe54010b8120c04a7a2dcfc8119227).
[`manifest.json`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/manifest.json)
records the study configuration, every declared margin, the cleverly version and commit the run was
produced from, the interpreter and library versions, and a hash of every published result and of
every study module.

Replication seeds are derived per scenario and replication rather than from the total, so
replication *k* is a fixed sample whatever the study's size. That is what lets the fast test suite
redraw committed replications, refit them, and require the published rows back -- the check that
keeps these artefacts evidence about the current code rather than a record of an old run.

Each registered study derives its seeds from the seed on its own record, so the rows in this
document sample different sets of datasets and their verdicts are independent draws rather than
one draw reported several times. A test requires that: it redraws each study's first
replication through its own runner and refuses two studies that produce the same sample.

The public R specifications are used directly except for ATT and ATC, which use the constrained,
one-dimensional updater exercised in the pinned package's own tests. The public ATT convenience
path fails to converge on a small fraction of bounded-continuous samples; the package-tested path
is fixed for all replications, and no failed sample is discarded or replaced. The R side aborts on
any failed replication rather than dropping it, and the property study refuses to summarize a run
that lost one.

The study does not establish parity for cleverly's default cross-fitting, repeated cross-fitting,
CV-TMLE evaluation, simultaneous intervals, bootstrap inference, missing outcomes, weights,
clustering, strata, multi-valued treatment, or flexible learners. Each is a distinct method or
composition and needs its own evidence row.

## Stacked point-treatment CV-TMLE

This row covers the default cross-validated point-treatment construction described by Levy
(2018): nuisance predictions are out of fold, one targeting regression is fitted over the stacked
validation rows, and the updated regression is evaluated over the whole sample.  Zheng & van der
Laan (2011) supplies the wider CV-TMLE framework; the source boundary is mapped in
[Ordinary TMLE and CV-TMLE](estimator-variants.md#ordinary-tmle-and-cv-tmle).

The bounded implementation witness compares cleverly with R
[`tmle3`](https://github.com/tlverse/tmle3) 0.2.0 at commit
[`ed72f8a`](https://github.com/tlverse/tmle3/tree/ed72f8a20e64c914ab25ffe015d865f7a9963d27)
and [`sl3`](https://github.com/tlverse/sl3) at commit
[`0e8f236`](https://github.com/tlverse/sl3/tree/0e8f2365bcbe54010b8120c04a7a2dcfc8119227).
Python generates every realized sample and its treatment-stratified ten-fold assignment.  R
reconstructs those exact validation indices with `origami`, asserts them on the `tmle3` task,
wraps corresponding GLMs in `Lrnr_cv`, and uses `tmle3_Update(cvtmle = TRUE)`.  The two sides also
share the 0.025--0.975 propensity bounds, treatment contrast, covariates, pointwise 95% interval
level, and targeting tolerances that affect the comparison.

The binary law covers `ey1`, `ey0`, `ate`, `att`, `atc`, `ey_obs`, `par`, `paf`, `rr`, and `or`;
the bounded-continuous law covers the same catalog except its binary-only PAF, RR, and OR.  Every
implementation-estimand cell is tested independently against exact truth before the paired tests,
so agreement between two poor fits cannot earn the row.  The R runner aborts the whole study on
any failed fit, changed fold, missing estimand, or silently dropped replication.

### Measured values

Names beginning `margin:` are the thresholds declared *before* the run; everything else is measured
from the committed results. Both are resolved by name and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | replications per law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 34 | implementation-estimand tests against truth |
| `independent_tests_passed` | 34 | of those, passing |
| `paired_tests_total` | 17 | scenario-estimand paired tests |
| `paired_tests_passed` | 17 | of those, passing |
| `property_cells_total` | 14 | repeated-sampling property cells |
| `property_cells_passed` | 14 | of those, passing |
| `max_standardized_bias` | 0.1082 | largest absolute bias in empirical standard deviations |
| `min_coverage` | 0.9375 | lowest measured coverage of a nominal 95% interval |
| `min_coverage_ci_lower` | 0.9203 | lowest exact 99% coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9329 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0884 | highest bootstrap SE-ratio endpoint |
| `max_se_ratio_resolution` | 0.0884 | farthest the widest SE-ratio interval reaches from 1.0 |
| `max_margin_utilization` | 0.6941 | largest share of a paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0128 | largest one-sided RMSE-ratio bound, against 1.10 |
| `min_coverage_difference_lower` | -0.0081 | smallest one-sided coverage-difference bound, against -0.025 |
| `max_calibration_excess_upper` | 0.0116 | largest SE-calibration-excess bound, against 0.05 |
| `properties[double_robustness/outcome_correct]:bias` | 0.000149 | bias with only the outcome nuisance correct |
| `properties[double_robustness/treatment_correct]:bias` | -0.0196 | bias with only the treatment nuisance correct |
| `properties[double_robustness/both_wrong]:bias` | -0.3348 | both-wrong negative control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5053 | fitted log-log sampling-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5384 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4737 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.5067 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 1.0077 | SE calibration where both nuisances are correct |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9738 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0450 | its 99% upper endpoint |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9550 | coverage of the same cell |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0325 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0627 | its 99% upper endpoint, against 0.10 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the positive control |
| `properties[crossfit_overfitting/stacked_cvtmle]:coverage` | 0.8950 | coverage with cross-fitted tree predictions |
| `properties[crossfit_overfitting/stacked_cvtmle]:se_ratio` | 0.9880 | SE calibration with cross-fitting |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.6500 | coverage with the deliberately in-sample tree |
| `properties[crossfit_overfitting/in_sample_control]:se_ratio` | 0.5792 | SE calibration of that control |
| `properties[crossfit_overfitting/stacked_cvtmle]:coverage_gain_ci_lower` | 0.1875 | paired 99% lower bound for coverage gained over the control |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval below |
| `margin:alpha` | 0.0500 | nominal size of the estimator's own intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the exact coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size the one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest RMSE ratio the one-sided upper bound may reach |
| `margin:coverage_noninferiority` | -0.0250 | smallest coverage difference the one-sided lower bound may reach |
| `margin:calibration_noninferiority` | 0.0500 | largest excess SE-calibration error the upper bound may reach |
| `margin:minimum_power` | 0.8000 | rejection lower bound the power control must clear |
| `margin:root_n_slope` | -0.5000 | the contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | the slower rate the interval must exclude |
| `margin:overfit_se_floor` | 0.8500 | SE ratio the cross-fit arm must restore |
| `margin:overfit_control_ceiling` | 0.7500 | ceiling the in-sample control's upper bound must stay below |
| `margin:overfit_coverage_gain` | 0.1500 | coverage cross-fitting must buy over the in-sample control |

### Comparison and property verdicts

Each implementation independently passes the ordinary study's 99% practical-bias, coverage, and
SE-calibration gates.  On the pairing, the 99% interval for the mean difference must lie inside
0.15 pooled empirical standard deviations.  Cleverly must also be non-inferior: the one-sided
RMSE-ratio upper bound is at most 1.10, the coverage-difference lower bound is at least -0.025, and
the excess absolute SE-calibration-error upper bound is at most 0.05 where the native inference
scales are comparable.  All 17 cells pass both the symmetric similarity and one-sided
non-inferiority claims, and both implementations remain independently valid.

The largest similarity-margin use occurs for continuous-law ATT, followed by ATC.  Both remain
inside the predeclared bound, and the RMSE, coverage, and calibration bounds remain well inside
their non-inferiority margins.  Binary-law `ey1` has the study's lowest coverage, at the same
`min_coverage` in both implementations, and its exact 99% lower endpoint still clears the 0.90
floor.  These are reported as the finite-sample cells that use the most evidence budget, not tuned
exceptions.

The 14 property cells use the same double-robustness, both-wrong discrimination, three-size
efficiency/calibration, empirical and reported root-n, interval-calibration, confounded-null, and
power instruments as the fold-evaluated row below.  They add the same paired flexible-tree
experiment: held-out
predictions must restore the SE ratio to its 0.85--1.20 band, the deliberately in-sample control's
upper bound must remain below 0.75, and the 99% lower bound for coverage gained by cross-fitting
must exceed 0.15.  The measured cross-fitted coverage is therefore evidence of relative recovery
and calibrated influence-curve scale, not a separate absolute 90% coverage claim; the primary GLM
study carries that gate.

PAF is the one inference-scale qualification.  Cleverly reports its fraction-scale influence
curve, while `tmle3` transforms a negative-log-complement/log-risk-ratio interval.  They target the
same parameter and their point performance and coverage are compared, but raw standard errors and
finite-sample endpoints on those different native scales are not declared numerically equivalent.
The study record names the exception and a test requires the scales actually to differ.

The committed [replicate results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/equivalence.csv),
[property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/properties.csv),
and [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/manifest.json)
record all rows, margins, seeds, configuration, source and reference hashes, package pins, and
result hashes.  The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/README.md)
gives the full and smoke regeneration commands.

This row is bounded to two complete-outcome point-treatment laws, corresponding GLM learners, one
ten-fold split, the declared truncation, a pooled update, whole-sample evaluation, and pointwise
Wald inference.  It does not establish repeated or nested cross-fitting, fold-evaluated or
fold-specific-epsilon CV-TMLE, simultaneous or bootstrap intervals, missing outcomes, weights,
clusters, strata, multi-valued treatment, broad learner-library selection, or severe
practical-positivity behavior.

## Fold-evaluated point-treatment CV-TMLE

This row covers cleverly's fold-evaluated CV-TMLE report: treatment-stratified ten-fold nuisance
fitting, one pooled targeting update, equal-fold plug-in evaluation, and the cross-validated
influence-curve variance.  It is kept separate from the stacked CV-TMLE row because averaging
fold reports rather than evaluating the updated regression over the whole sample is a genuine
finite-sample method choice.  The construction and its Zheng--van der Laan/Levy source boundary
are mapped in [Ordinary TMLE and CV-TMLE](estimator-variants.md#ordinary-tmle-and-cv-tmle).

The primary study uses the same binary and bounded-continuous laws as the ordinary TMLE study and
tests `ey1`, `ey0`, `ate`, `att`, and `atc` against exact truth.  The nuisance learners are
corresponding logistic and linear GLMs, the propensity is bounded to 0.025--0.975, and intervals
are pointwise 95% identity-scale Wald intervals.  There is no external implementation in this
row: a zero-row equivalence artifact records that absence instead of borrowing the stacked R
comparison.

### Measured values

Names beginning `margin:` are the thresholds declared *before* the run; everything else is measured
from the committed results. Both are resolved by name and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | replications per law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 10 | estimand-law tests against truth |
| `independent_tests_passed` | 10 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 14 | repeated-sampling property cells |
| `property_cells_passed` | 14 | of those, passing |
| `max_standardized_bias` | 0.0384 | largest absolute bias in empirical standard deviations |
| `min_coverage` | 0.9375 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9203 | lowest exact 99% coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9303 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0737 | highest bootstrap SE-ratio endpoint |
| `properties[double_robustness/outcome_correct]:bias` | 0.000149 | bias with only the outcome nuisance correct |
| `properties[double_robustness/treatment_correct]:bias` | -0.0196 | bias with only the treatment nuisance correct |
| `properties[double_robustness/both_wrong]:bias` | -0.3348 | both-wrong negative control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5053 | fitted log-log sampling-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5374 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4732 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.5071 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 1.0080 | SE calibration where both nuisances are correct |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9730 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0456 | its 99% upper endpoint |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9550 | coverage of the same cell |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0325 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0627 | its 99% upper endpoint, against 0.10 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the positive control |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:coverage` | 0.8950 | coverage with cross-fitted tree predictions |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:se_ratio` | 0.9910 | SE calibration with cross-fitting |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.6500 | coverage with the deliberately in-sample tree |
| `properties[crossfit_overfitting/in_sample_control]:se_ratio` | 0.5792 | SE calibration of that control |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:coverage_gain_ci_lower` | 0.1875 | paired 99% lower bound for coverage gained over the control |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval below |
| `margin:alpha` | 0.0500 | nominal size of the estimator's own intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the exact coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size the one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest RMSE ratio the one-sided upper bound may reach |
| `margin:coverage_noninferiority` | -0.0250 | smallest coverage difference the one-sided lower bound may reach |
| `margin:calibration_noninferiority` | 0.0500 | largest excess SE-calibration error the upper bound may reach |
| `margin:minimum_power` | 0.8000 | rejection lower bound the power control must clear |
| `margin:root_n_slope` | -0.5000 | the contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | the slower rate the interval must exclude |
| `margin:overfit_se_floor` | 0.8500 | SE ratio the cross-fit arm must restore |
| `margin:overfit_control_ceiling` | 0.7500 | ceiling the in-sample control's upper bound must stay below |
| `margin:overfit_coverage_gain` | 0.1500 | coverage cross-fitting must buy over the in-sample control |

### Statistical claims and controls

The first twelve property cells repeat the ordinary method's independent instruments: three
correct-nuisance combinations must have practically equivalent bias, both nuisances wrong must
be discriminated from that margin, efficiency and the coverage floor must hold at three sample
sizes, a correctly specified cell must have a two-sided calibrated SE ratio and coverage, and a
confounded sharp null is paired with a nonzero-effect power control.

Both root-n rates -- the empirical spread and the mean reported standard error -- use the same
predeclared equivalence band: the fitted 99% interval must lie within 0.125 of -1/2, half the
distance to the -1/4 alternative it must also exclude.  A band rather than containment of -1/2,
and the same band for both statistics rather than one rule each.  Requiring an increasingly
precise Monte Carlo interval to contain exactly -1/2 would eventually reject the arithmetic
scaling for a negligible finite-sample remainder: at these replication counts the reported-SE
rate is already resolved to about a thousandth, and every one of the five studies would fail
that rule today.

Two additional cells exercise the reason to cross-fit.  A fully grown regression tree is fitted
on the nonlinear law, once with held-out nuisance predictions and once as a deliberately
in-sample control on the identical 400 samples of size 500.  The cross-fitted report must keep its
99% SE-ratio interval within 0.85--1.20, the control's upper endpoint must remain below 0.75, and
the paired 99% lower endpoint for the coverage gain must clear 0.15.  The cross-fitted cell's
measured coverage is not presented as a separate 90% validity claim: its evidence is restored SE
calibration and a load-bearing improvement over the overfit control, while the primary GLM study
carries the absolute coverage gate.

The committed [replicate results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/replicates.csv.gz),
[property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/properties.csv),
and [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/manifest.json)
record the primary and control samples, every margin and seed, the exact estimator configuration,
source hashes, and result hashes.  The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/README.md)
gives the regeneration command.

This study does not establish external parity, repeated or nested cross-fitting, a fold-specific
targeting epsilon, simultaneous or bootstrap intervals, missing outcomes, weights, clusters,
strata, multi-valued treatment, ratio estimands, observed-risk functionals, or behavior under
severe practical-positivity violations.  It tests one fixed ten-fold assignment per sample and
the declared complete-outcome point-treatment laws.

## Selector-based point-treatment C-TMLE

This row separates three selector constructions from the outcome-adaptive construction below.
The bounded implementation witness compares Cleverly's greedy, ordered, and discrete paths with
R [`ctmle`](https://github.com/jucheng1992/ctmle) 0.1.2 at pinned commit
[`18de559`](https://github.com/jucheng1992/ctmle/tree/18de559f47dc1286617350a0668391e80e1dbf7c).
That package is the maintained comparator for these selector entry points; it is not a tlverse
package. The tlverse comparison applies only to OAT in the next section.

Python generates every binary-outcome sample, its exact ATE, and the treatment-stratified
five-fold selector assignment, taken off the Cleverly fit that produced the subject's own row
and supplied unchanged to R, which asserts it is a partition of the sample before selecting
against it. Both sides use all three DGP covariates,
corresponding logistic GLMs, 0.025--0.975 propensity bounds, pointwise 95% intervals, and the
unpenalized selector loss. Cleverly's default penalty follows the published trace-plus-bias
criterion and is not presented as numerical parity with R's implementation-specific adjustment.
The parity fit also disables cross-fitting; the independent property study exercises a public
nested-cross-fit configuration instead -- five outer folds, three selection folds and two inner
folds, with the penalty on. Those fold counts are not the shipped defaults (ten, five and two),
so what the property cells establish is that nested cross-fitting works, not that the default
fold counts were the ones measured.

### Measured values

Names beginning `margin:` are declared rules; all other values are computed from the committed
artifacts. The documentation test resolves each name and checks the printed rounding.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | replications per selector strategy |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 6 | implementation-strategy tests against truth |
| `independent_tests_passed` | 6 | of those, passing |
| `paired_tests_total` | 3 | paired selector-strategy tests |
| `paired_tests_passed` | 3 | of those, passing |
| `property_cells_total` | 14 | independent property cells |
| `property_cells_passed` | 14 | of those, passing |
| `max_standardized_bias` | 0.0426 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9387 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9136 | lowest exact 99% coverage endpoint |
| `min_se_ratio_ci_lower` | 0.8850 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0342 | highest bootstrap SE-ratio endpoint |
| `max_margin_utilization` | 0.0189 | largest share of the paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0051 | largest paired RMSE-ratio upper bound |
| `min_coverage_difference_lower` | -0.0063 | smallest paired coverage-difference lower bound |
| `max_calibration_excess_upper` | 0.0122 | largest paired SE-calibration-excess upper bound |
| `properties[double_robustness/both_correct]:standardized_bias` | 0.0155 | both nuisances correct |
| `properties[double_robustness/outcome_correct]:standardized_bias` | -0.0242 | only the outcome nuisance correct |
| `properties[double_robustness/treatment_correct]:standardized_bias` | 0.1408 | only the treatment nuisance correct |
| `properties[double_robustness/treatment_correct]:n` | 2000 | observations that leg needs to resolve its remainder |
| `properties[double_robustness/both_wrong]:standardized_bias` | 0.8464 | both-wrong negative control |
| `properties[selector_necessity/collaborative]:rmse_ratio` | 0.1077 | collaborative RMSE divided by the empty-path control RMSE |
| `properties[selector_necessity/collaborative]:se_ratio` | 1.2539 | reported SE over empirical spread in that cell -- see below |
| `properties[root_n_rate/empirical_sd]:slope` | -0.4954 | fitted empirical-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5270 | its 99% lower endpoint, against a band of -0.625 to -0.375 |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4640 | its 99% upper endpoint, which must also exclude -0.25 |
| `properties[root_n_rate/reported_se]:slope` | -0.5030 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9437 | calibration-cell coverage |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 0.9942 | calibration-cell SE ratio |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9579 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0340 | its 99% upper endpoint |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0275 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0561 | its 99% upper endpoint, against 0.05 + 0.05 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the power control |
| `margin:confidence_level` | 0.9900 | confidence level of Monte Carlo intervals |
| `margin:alpha` | 0.0500 | nominal estimator size |
| `margin:nominal_coverage` | 0.9500 | nominal estimator coverage |
| `margin:bootstrap_replicates` | 10000 | resamples per bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor for the exact coverage lower endpoint |
| `margin:over_coverage_ceiling` | 0.9900 | coverage above this is labeled conservative |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | one-sided type-I error ceiling |
| `margin:paired_difference` | 0.1500 | paired mean-difference margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | paired RMSE-ratio upper limit |
| `margin:coverage_noninferiority` | -0.0250 | paired coverage-difference lower limit |
| `margin:calibration_noninferiority` | 0.0500 | paired calibration-excess upper limit |
| `margin:minimum_power` | 0.8000 | power-control rejection lower bound |
| `margin:root_n_slope` | -0.5000 | predicted root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:selector_rmse_ratio` | 0.5000 | maximum collaborative-to-empty RMSE ratio |

### Statistical claims and limitations

The robustness cells use one confounded linear law and four explicit nuisance configurations.
Both-correct, outcome-correct-only, and treatment-correct-only bias intervals must fit inside the
same equivalence margin; the both-wrong control must be discriminated outside it. The selector
instrument makes selection necessary by fitting a constant outcome regression on an
instrument/confounder law. The public collaborative search must beat a selector restricted to
the empty propensity path by the declared RMSE ratio, so a hard-coded empty selector cannot pass.
The remaining cells check empirical and reported root-n contraction, efficiency at three sample
sizes, two-sided interval calibration, type-I error, and a power positive control.

What the forced-selection cell claims is the RMSE ratio, and only that. Its reported standard
error is `properties[selector_necessity/collaborative]:se_ratio` of the empirical spread --
conservative, and outside the 0.80--1.20 screen the primary cells answer to. No SE or coverage
gate is applied to it, which is deliberate rather than an oversight: an instrument/confounder
law with a constant outcome regression is built to make *selection* necessary, not to be a
setting where the influence-curve variance is the efficient one. It is worth stating plainly
because the consequence is a real gap in this row -- `interval_calibration` is measured where
both nuisances are correct and the search therefore has nothing to do, so no cell in this study
asks for calibrated inference *while* selection is load-bearing.

The committed [replicate results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/equivalence.csv),
[property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/properties.csv),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/manifest.json),
and [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/README.md)
record the results, provenance, and regeneration commands.

Three qualifications belong on the comparison itself rather than in a CSV.

**Two of the three strategies reach the same R entry point.** R `ctmle` has one pre-ordered
selector, so the `ordered` and `discrete` cells are both compared against
`ctmleDiscrete(preOrder = TRUE)`. The correspondence is earned rather than assumed: the
`discrete` cell's candidate list is exactly the nested prefix ladder that mode enumerates. It
follows that an arbitrary candidate list has no reference here, and that the row carries two
reference constructions rather than three, on three separate draws.

**The two sides select differently even where they agree.** Cleverly refits the outcome
regression inside two nested folds within each selection fold; R scores every fold against one
full-sample `Q`. The paired cells use 0.019 of the similarity margin at their widest, so the
reports agree -- but on a law where the selector's choice is stable, which is what makes
agreement here evidence about the C-TMLE machinery rather than about the selection rule. That
the search itself is load-bearing is established by `selector_necessity` below and by the unit
tests, not by this comparison.

**One robustness cell is sized differently from its siblings.** `treatment_correct` runs at
`properties[double_robustness/treatment_correct]:n` observations where the other three run at
700. It is the leg that leans on inverse weighting, and at 700 its `O(n^-1)` remainder is about
0.28 empirical standard deviations -- outside the margin, for a reason that is not first-order
bias. Raising `n` for that cell resolves the remainder against an unchanged margin; the margin
was not moved after seeing it.

The R parity claim is binary, two-arm, complete-outcome, GLM, non-cross-fitted, unpenalized ATE
only. Continuous outcomes are covered by independent implementation tests elsewhere, not this
parity witness. The row does not establish parity for Cleverly's default penalty or nested
cross-fitting, ratios or arm means, missing outcomes, weights, clusters, strata, multi-valued
treatment, flexible learner libraries, simultaneous or bootstrap intervals, or severe
practical-positivity behavior.

Two coverage gaps are worth naming rather than leaving to be inferred. The `ordered` cell pins
an explicit covariate order, so the *default* `preorder="logistic"` ordering -- what
`strategy="ordered"` does when the caller supplies nothing -- is exercised by neither half of
this row. And the property cells all run the default `greedy` search, so `ordered` and
`discrete` have parity evidence without repeated-sampling evidence of their own.

## Outcome-adaptive point-treatment C-TMLE

This row covers `CTMLE(strategy="oat")`, whose treatment mechanism is fitted on the vector of
arm-specific outcome-regression predictions rather than on a selected subset of the original
covariates. The bounded implementation witness uses archived tlverse
[`ctmle3`](https://github.com/tlverse/ctmle3) 0.1.0 at commit
[`a4ea77b`](https://github.com/tlverse/ctmle3/tree/a4ea77b07747dfee9b2eecb9cbca88262e0559ea),
with contemporaneous `tmle3` at
[`3a61005`](https://github.com/tlverse/tmle3/tree/3a610058cd89c17bb417c15fc891254388787f33) and `sl3` at
[`821ca89`](https://github.com/tlverse/sl3/tree/821ca890cb8701fdb59f823e28c6356e50d092bc).

Both sides receive the same binary samples and exact truths, fit the same three-covariate
logistic outcome regression, and use the archived non-cross-fitted OAT construction. The report
checks both treatment-specific means, ATE, marginal risk ratio, and marginal odds ratio. Ratio
standard errors and intervals use their log-scale delta-method curves. Cleverly's public
cross-fitted generated-regressor behavior is tested independently rather than attributed to the
archived implementation.

### Measured values

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | replications of the binary law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 10 | implementation-estimand tests against truth |
| `independent_tests_passed` | 10 | of those, passing |
| `paired_tests_total` | 5 | paired estimand tests |
| `paired_tests_passed` | 5 | of those, passing |
| `property_cells_total` | 14 | independent property cells |
| `property_cells_passed` | 14 | of those, passing |
| `max_standardized_bias` | 0.0513 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9425 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9179 | lowest exact 99% coverage endpoint |
| `min_se_ratio_ci_lower` | 0.9211 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0769 | highest bootstrap SE-ratio endpoint |
| `max_margin_utilization` | 0.0016 | largest share of the paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0005 | largest paired RMSE-ratio upper bound |
| `min_coverage_difference_lower` | 0 | smallest paired coverage-difference lower bound |
| `max_calibration_excess_upper` | 0.000592 | largest paired SE-calibration-excess upper bound |
| `properties[robustness_contract/outcome_correct]:standardized_bias` | 0.0166 | outcome-correct OAT bias |
| `properties[robustness_contract/outcome_wrong]:standardized_bias` | -2.4768 | outcome-wrong negative control |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5287 | 99% lower endpoint of the empirical-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4651 | its upper endpoint, which must also exclude -0.25 |
| `properties[root_n_rate/empirical_sd]:slope` | -0.4967 | fitted empirical-spread rate |
| `properties[root_n_rate/reported_se]:slope` | -0.5022 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9467 | calibration-cell coverage |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 0.9870 | calibration-cell SE ratio |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9506 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0250 | its 99% upper endpoint |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0725 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0994 | its 99% upper endpoint, against 0.05 + 0.05 |
| `properties[type_i_error/sharp_null]:coverage_ci_lower` | 0.9006 | its exact 99% coverage endpoint, against a floor of 0.90 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the power control |
| `properties[crossfit_overfitting/cross_fitted_oat]:coverage` | 0.9325 | coverage with cross-fitted tree predictions |
| `properties[crossfit_overfitting/cross_fitted_oat]:standardized_bias` | -0.4361 | that arm's bias, in empirical standard deviations |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.6225 | coverage with the in-sample tree control |
| `properties[crossfit_overfitting/cross_fitted_oat]:se_ratio` | 1.0681 | SE ratio with cross-fitting |
| `properties[crossfit_overfitting/in_sample_control]:se_ratio` | 0.5565 | SE ratio without cross-fitting |
| `properties[crossfit_overfitting/cross_fitted_oat]:coverage_gain_ci_lower` | 0.2525 | paired 99% lower bound for coverage gained by cross-fitting |
| `properties[generated_design/oracle_design]:se_ratio` | 0.9892 | SE ratio with the design pinned at the truth |
| `properties[generated_design/oracle_design]:se_ratio_ci_lower` | 0.9396 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[generated_design/oracle_design]:se_ratio_ci_upper` | 1.0430 | its 99% upper endpoint |
| `properties[generated_design/estimated]:se_ratio` | 0.9525 | the same ratio with the design estimated |
| `properties[generated_design/estimated]:se_ratio_ci_lower` | 0.9042 | its 99% lower endpoint |
| `properties[generated_design/estimated]:se_ratio_ci_upper` | 1.0047 | its 99% upper endpoint |
| `properties[generated_design/estimated]:se_ratio_deficit_lower` | -0.0546 | paired 99% lower endpoint for estimated minus pinned |
| `properties[generated_design/estimated]:se_ratio_deficit_upper` | -0.0191 | its upper endpoint, which must clear the floor below |
| `margin:confidence_level` | 0.9900 | confidence level of Monte Carlo intervals |
| `margin:alpha` | 0.0500 | nominal estimator size |
| `margin:nominal_coverage` | 0.9500 | nominal estimator coverage |
| `margin:bootstrap_replicates` | 10000 | resamples per bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor for the exact coverage lower endpoint |
| `margin:over_coverage_ceiling` | 0.9900 | coverage above this is labeled conservative |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | one-sided type-I error ceiling |
| `margin:paired_difference` | 0.1500 | paired mean-difference margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | paired RMSE-ratio upper limit |
| `margin:coverage_noninferiority` | -0.0250 | paired coverage-difference lower limit |
| `margin:calibration_noninferiority` | 0.0500 | paired calibration-excess upper limit |
| `margin:minimum_power` | 0.8000 | power-control rejection lower bound |
| `margin:root_n_slope` | -0.5000 | predicted root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:overfit_se_floor` | 0.8500 | cross-fit SE-ratio lower limit |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample SE-ratio upper limit |
| `margin:overfit_coverage_gain` | 0.1500 | cross-fit coverage-gain lower limit |
| `margin:generated_design_deficit` | 0.0100 | smallest paired SE-ratio deficit the control must establish |

### Statistical claims and limitations

OAT deliberately has a narrower robustness contract than selector C-TMLE. With the outcome
regression correct, its bias interval must fit inside the same equivalence margin; with that
regression wrong, the negative control must be discriminated outside it. No treatment-correct-only
claim is made because OAT's mechanism is a projection on the generated outcome-regression design,
not a fit of treatment on the original covariates. Root-n, efficiency, calibration, null, and power
cells check the remaining first-order report. A fully grown tree then makes overfitting visible:
cross-fitted OAT must restore the SE ratio, the in-sample control must retain its underestimated
spread, and the paired coverage-gain lower bound must clear the declared threshold.

That cross-fitted arm's measured coverage is evidence of *relative* recovery and of a calibrated
influence-curve scale, not a separate absolute coverage claim: a fully grown tree on this law
carries `properties[crossfit_overfitting/cross_fitted_oat]:standardized_bias` empirical standard
deviations of nuisance bias, which is why the cell is gated on its SE ratio and on the paired
gain rather than on the coverage floor. The primary GLM study carries the absolute gate.

### What the reported interval leaves out

OAT fits the treatment mechanism on `Qbar` itself, so when `Qbar` is estimated the *model
class* `g` is chosen from is random too, and the influence curve does not see that. The
`generated_design` cells measure the consequence directly: one law, one set of draws, and a
single difference between the two cells -- whether `Qbar` moves.

With the design pinned at the truth the SE ratio is
`properties[generated_design/oracle_design]:se_ratio`, with a 99% interval from
`properties[generated_design/oracle_design]:se_ratio_ci_lower` to
`properties[generated_design/oracle_design]:se_ratio_ci_upper`, inside the calibration band.
With it estimated the ratio is `properties[generated_design/estimated]:se_ratio`. **Neither
interval on its own excludes 1**, and that is the honest statement about absolute calibration
at this budget: an SE ratio's Monte Carlo error is dominated by the empirical spread in its
denominator, which at these replication counts is worth about two percent on its own.

The *paired* difference is what resolves, because the two cells share their draws and that
common denominator error cancels rather than being counted twice. It runs from
`properties[generated_design/estimated]:se_ratio_deficit_lower` to
`properties[generated_design/estimated]:se_ratio_deficit_upper`, entirely below zero. So the
omission is real, and it is worth a few percent of a reported standard error -- not a defect
that shows up as invalid coverage. The primary cells, the calibration cell and the coverage
floor all clear their gates, and this row does not claim otherwise in either direction.

The margin on that control is a floor on a defect rather than a tolerance for one. If the
reported covariance is ever made to carry this term, the control stops being discriminated and
this row goes red -- which is the correct signal that the limitation documented here has gone
stale, rather than a regression.

The wider contract is the ordinary one and is worth restating where a reader meets it: what is
reported is the cross-fitted EIF covariance evaluated at the estimated nuisances. It does not
add the adaptive-`g` influence term from the collaborative-double-robust theorem, and the
cells above are what put a measured size on one part of that gap rather than leaving it as a
caveat. A nonparametric bootstrap reruns the whole construction and so carries the omitted
terms, but it was measured on this law and did not improve calibration over the reported
interval, so it is not presented here as a remedy.

One configuration difference is declared rather than hidden. Cleverly bounds the propensity to
0.025--0.975 and the archived R path applies no truncation of its own, so the manifest's
`g_bounds` entry describes the subject's setting and not a shared one. On this law the true
propensity spans 0.085 to 0.904, so the bound is never active and no paired comparison is
affected by it -- which is the reason it is recorded as a difference rather than treated as a
reason to exclude a cell.

The committed [replicate results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/equivalence.csv),
[property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/properties.csv),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/manifest.json),
and [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/README.md)
record the results, provenance, and regeneration commands.

The parity claim is binary, two-arm, complete-outcome, GLM, and non-cross-fitted. During
feasibility work, the archived stack failed the analogous continuous law because its length-two
outcome bounds enter a scalar `if` condition; the runner treats that as a reference limitation,
not a dropped replication. The row does not establish continuous parity, multi-arm parity,
missing outcomes, weights, clusters, strata, simultaneous or bootstrap intervals, broad learner
libraries, or severe practical-positivity behavior. Cross-fitted public behavior is supported by
the property study, not by the archived parity result.

## End-of-study longitudinal TMLE

This row covers the two-time-point intervention-specific mean identified by the sequential
g-formula, with monotone censoring and both static and dynamic plans. The parameter, longitudinal
double-robustness claim, and efficient influence curve follow Bang and Robins (2005), van der Laan
and Gruber (2012), and Petersen et al. (2014). The implementation comparison uses the canonical R
[`ltmle`](https://www.jstatsoft.org/article/view/v081i01) package at version 1.3-0; agreement with R
is secondary to the finite-support functional and Gateaux EIF in
[`tests/discrete_law_longitudinal.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/discrete_law_longitudinal.py).

### Matched implementation comparison

Python draws 1,600 censored samples and hands the same rows to cleverly and R. Each implementation
reports the means under never treatment, always treatment, and the dynamic plan “treat, then
continue if L2 is positive,” plus the always-minus-never and dynamic-minus-never contrasts. Both
receive the generating treatment and censoring probabilities, follower-stratified quasibinomial
sequential regressions, nonbinding cumulative-g bounds, pointwise 95% intervals, and influence-curve
variance. R is invoked once per regimen; contrast standard errors use the difference of the two
rowwise influence curves, preserving covariance.

The paired mean discrepancies are numerical-solver scale rather than statistical scale: the
largest share of the declared similarity margin used is `max_margin_utilization`. Each
implementation is also tested against quadrature truth on its own, and both verdicts are carried
beside the paired one rather than folded into it; a regeneration fails if either is false. Note
that the quadrature truth the R rows are scored against comes from this package's own
`cleverly.datasets.longitudinal`, so the reference's truth column is not independent of this
codebase. What makes it usable is that the quadrature is checked separately, against Monte Carlo
and by node refinement, in `tests/unit/test_datasets_longitudinal.py`.
The [replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle/replicates.csv.gz),
[paired verdicts](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle/equivalence.csv),
and [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle/manifest.json)
carry the samples' results, package checksum, R image digest, source commit, formulas, seeds, and
artifact hashes.

### What the agreement does and does not witness

Agreement this close is a strong statement about the sequential regressions, the follower and
censoring masks, the cumulative mechanism product, and the influence curve. It is a weaker
statement about the targeting step than five passing comparisons suggest, and the section says so
rather than leaving a reader to infer it.

Two measurements. First, `initial_estimate` is the earlier node's regression of the *already
targeted* later node — R's `fit$Q[[1]]` regresses the updated `Q.kplus1`, and cleverly's first
step does the same — so `max_targeting_displacement` and `median_targeting_displacement`, the
distance from the plug-in in units of the estimate's own standard error, measure the final
fluctuation only. Second, and this is the substantive one: a plug-in built from the same two
regressions with *no* fluctuation at either node still clears both acceptance gates on
`ey_regimen[never]`, `ey_regimen[always]`, and `ate_regimen[always vs never]`. Only the two
estimands carrying the dynamic rule separate it.

So the targeting evidence in this row is the dynamic plan and the `targeting_necessity` property
cells below, not the count of paired comparisons.
`tests/e2e/test_ltmle_targeting_slow.py` builds that untargeted plug-in and asserts exactly which
estimands survive, so the limitation is measured on every run rather than described once.

### Independent statistical properties

The property study samples the exact binary support law rather than the continuous comparison law.
Its longhand functional supplies exact static and dynamic truths, and its Gateaux derivative supplies
the efficiency bound without reading either estimator. The double-robust cells fit all sequential Q
regressions correctly, all treatment-and-censoring regressions correctly, both, or neither. The first
three must place their 99% bias interval inside 0.25 empirical standard deviations; both-wrong must
place it outside by the same instrument.

The `targeting_necessity` cells hold the outcome model at a constant and the mechanism at the
saturated fit, which is the configuration where the fluctuation does all the adjusting. Each
contrast is reported twice off the same draw: as the estimator computes it, and as the identical
backward recursion computes it with no fluctuation at either node. The targeted arm must place its
bias inside the equivalence margin and the plug-in must place its outside, and the pair must
additionally move apart by at least `margin:targeting_displacement` empirical standard deviations
— without that clause a targeting step that did nothing would satisfy both arms by making them the
same number.

Root-n contraction is measured across n=500, 2,000, and 8,000 for both contrasts. The n=500 cells
are controls whose inference the row does not claim: each must *resolve*, placing its exact 99%
coverage interval clear of nominal on one side or the other. Below is a small-sample limitation
established and published; at or above the floor is the estimator turning out to be adequate there,
which is a result rather than a failure. Only an interval straddling nominal fails, because it is
the one outcome that says nothing. At n=2,000 and n=8,000 the positive inference cells clear the
coverage floor, while the three-size empirical and reported-SE slopes satisfy the root-n margin
and exclude -1/4. The dedicated n=2,000 calibration cells additionally compare empirical and reported
spread with the exact EIF bound; all three ratios are resampled on one set of draws, so the
identity between them holds interval by interval. Multiplying standard errors by 0.70 and adding one
efficiency-bound unit of independent noise are registered controls and must fail in their predicted
directions.

The sharp-null law replaces the outcome probabilities on the cells the contrasted plans traverse
and nothing else, so it shares this law's treatment, censoring, and L2 mechanisms exactly. L2 still
moves the outcome from 0.25 to 0.75, censoring is still informative through it, and the first arm
still matters, so the null is one an estimator has to work for: a baseline-only standardisation
returns -0.0088 rather than the truth. Both contrasts are exactly zero under it; only the static
one is registered as a cell, because a type-I cell needs a nonzero-effect power control and the
dynamic contrast's power at n=4,000 is near 0.43 against a floor of 0.80. The alternative is the
original finite law, whose static contrast is exactly 0.15625. The type-I cell is a calibration
claim about the adjusted test; discrimination between a correct and an incorrect adjustment lives
in the double-robustness controls, not here.

Against this harder null the test over-rejects mildly, and the row says so rather than reporting
only that its gate passed. The rate is 0.0700 at n=4,000 on 800 replications, about 2.6 Monte
Carlo standard errors above nominal, and coverage in the same cell is 0.9300. The one-sided bound
of 0.0965 clears the predeclared ceiling of 0.1000, so the cell passes, but it passes with roughly
three rejections to spare and the point estimate is genuinely above 0.05. This is the same
finite-sample story the n=500 control tells: the influence-curve standard error is slightly
optimistic here, and it is being published rather than absorbed. The previous, degenerate null
reported 0.0575 — lower, and about a law that required no longitudinal adjustment at all.

### Measured values

Names beginning `margin:` are predeclared thresholds; all other values are resolved from the committed
artifacts and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_total` | 10 | implementation-estimand truth tests |
| `independent_tests_passed` | 10 | truth tests passing |
| `paired_tests_total` | 5 | paired estimand comparisons |
| `paired_tests_passed` | 5 | paired comparisons passing |
| `property_cells_total` | 30 | independent property cells |
| `property_cells_passed` | 30 | property cells passing |
| `max_standardized_bias` | 0.0180 | largest primary standardized bias |
| `min_coverage` | 0.9387 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9216 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9290 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0641 | highest primary SE-ratio endpoint |
| `max_margin_utilization` | 4.449e-08 | largest share of paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0000 | largest paired RMSE-ratio bound |
| `min_coverage_difference_lower` | 0 | smallest paired coverage-difference bound |
| `max_calibration_excess_upper` | 1.969e-08 | largest paired calibration-excess bound |
| `properties[double_robustness/static__both_wrong]:standardized_bias` | -0.4195 | static both-wrong control |
| `properties[double_robustness/dynamic__both_wrong]:standardized_bias` | 0.5944 | dynamic both-wrong control |
| `properties[root_n_and_efficiency/static__n_500]:coverage` | 0.8988 | static small-sample control coverage |
| `properties[root_n_and_efficiency/dynamic__n_500]:coverage` | 0.9113 | dynamic small-sample control coverage |
| `properties[interval_calibration/static__correctly_specified]:efficiency_empirical_ratio` | 0.9948 | static empirical spread over exact EIF bound |
| `properties[interval_calibration/dynamic__correctly_specified]:efficiency_empirical_ratio` | 1.0093 | dynamic empirical spread over exact EIF bound |
| `properties[type_i_error/static__sharp_null]:rejection_rate` | 0.0700 | confounded-null rejection rate |
| `properties[type_i_error/static__sharp_null]:rejection_ci_upper` | 0.0965 | one-sided bound the type-I ceiling is checked against |
| `properties[power/static__alternative]:rejection_rate` | 0.9700 | alternative rejection rate |
| `properties[root_n_and_efficiency/static__n_2000]:coverage_ci_lower` | 0.9020 | tightest primary property coverage endpoint |
| `properties[targeting_necessity/static__targeted]:standardized_bias` | 0.0435 | static contrast as the estimator computes it |
| `properties[targeting_necessity/static__untargeted]:standardized_bias` | -0.3447 | the same recursion with no fluctuation |
| `properties[targeting_necessity/dynamic__untargeted]:standardized_bias` | 0.8970 | dynamic contrast with no fluctuation |
| `properties[targeting_necessity/static__targeted]:targeting_displacement` | 0.3909 | least-displaced contrast, in targeted standard deviations |
| `max_targeting_displacement` | 0.0938 | largest final-fluctuation move, in standard errors |
| `median_targeting_displacement` | 0.0117 | median final-fluctuation move, in standard errors |
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
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound |
| `margin:minimum_power` | 0.8000 | power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope upper bound |
| `margin:excluded_slope` | -0.2500 | rate the interval must exclude |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.2500 | least the fluctuation must move the estimate |

### Claim boundary

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, and conditionally independent censoring. Single-correct-nuisance cells establish
consistency only; calibrated influence-curve inference is claimed where both nuisance sequences are
correct.

Two limits are internal to the evidence rather than to the estimator. The targeting step is
witnessed by the dynamic plan and by the `targeting_necessity` cells, not by the three static
paired comparisons, which an untargeted plug-in also passes. And positivity is comfortable
throughout: the smallest cumulative mechanism product on the comparison law sits between 0.006 and
0.03, and the property law bounds every conditional into [0.25, 0.75], so no cell here speaks to
near-positivity behaviour or to an active bound.

This row excludes survival, competing risks, longitudinal MSMs, observation weights,
clustering, simultaneous bands, flexible learning, cross-fitting, active truncation, and R parity
for learned mechanisms. Those are different estimators or compositions and require their own rows.
