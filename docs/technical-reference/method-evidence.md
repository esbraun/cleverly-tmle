# Method evidence studies

Each section here is one row of the [method evidence grid](../evidence.md#method-evidence-grid).
A row exists only for a study registered in `tests/studies/evidence/registry.py`; the shared
machinery in `tests/studies/evidence/` computes every verdict and every number quoted below, and
`tests/unit/test_method_evidence.py` checks this document against the committed results rather
than trusting it. See [adding a method row](../evidence.md#adding-a-method-row).

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

Every figure this section quotes is resolved from the committed results by name and checked at the
precision it is printed to, so a stale or mistyped number is a test failure rather than a reading
error.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | replications per law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 34 | implementation-estimand tests against truth |
| `independent_tests_passed` | 34 | of those, passing |
| `paired_tests_total` | 17 | scenario-estimand cells compared with `tmle3` |
| `paired_tests_passed` | 17 | of those, passing |
| `property_cells_total` | 11 | repeated-sampling property cells |
| `property_cells_passed` | 11 | of those, passing |
| `max_standardized_bias` | 0.1254 | largest bias, in empirical standard deviations |
| `min_coverage` | 0.9344 | lowest measured coverage of a nominal 95% interval |
| `min_coverage_ci_lower` | 0.9168 | lowest exact 99% coverage endpoint, against a floor of 0.90 |
| `min_se_ratio_ci_lower` | 0.9041 | lowest bootstrap SE-ratio endpoint, against a band of 0.80--1.20 |
| `max_se_ratio_ci_upper` | 1.0639 | highest bootstrap SE-ratio endpoint |
| `max_se_ratio_resolution` | 0.0959 | how far from 1.0 the widest SE-ratio interval still reaches |
| `max_margin_utilization` | 0.7445 | largest share of the paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0087 | largest one-sided RMSE-ratio bound, against a margin of 1.10 |
| `min_coverage_difference_lower` | -0.0088 | smallest one-sided coverage-difference bound, against -0.025 |
| `max_calibration_excess_upper` | 0.0128 | largest comparable SE-calibration-excess bound, against 0.05 |
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
| `properties[root_n_rate/empirical_sd]:slope` | -0.5429 | fitted log-log contraction rate of the sampling spread |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.6046 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4826 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.4974 | the same rate for the mean reported standard error |
| `properties[root_n_and_efficiency/n_500]:se_ratio` | 0.9433 | SE calibration at n = 500 |
| `properties[root_n_and_efficiency/n_2000]:se_ratio` | 0.9913 | SE calibration at n = 2,000 |
| `properties[root_n_and_efficiency/n_8000]:se_ratio` | 1.0701 | SE calibration at n = 8,000 |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0325 | rejection under a confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0627 | its 99% upper endpoint, against 0.05 + 0.05 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under a real effect -- the positive control |

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
than about a tenth is not something this study can rule out. Coverage carries the validity claim
instead, where the exact interval is tight enough to separate the nominal rate from a two-point
shortfall.

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
record four nuisance-model cells, three sample sizes, a fitted rate, a sharp-null experiment and a
power control:

| claim | positive evidence | negative control or discriminator |
| --- | --- | --- |
| double-robust consistency | with both nuisances correct, with only the outcome regression correct, and with only the treatment mechanism correct, the 99% interval for the bias lies inside 0.25 empirical standard deviations of zero | with both nuisances wrong the same interval must lie entirely *outside* that margin -- the same instrument in both directions, so a study too small to say anything fails both halves rather than passing the first |
| root-n behaviour | the log empirical spread of the estimates, regressed on log `n` across three sample sizes, has a 99% interval containing -1/2 | that interval must also exclude -1/4, so a merely decreasing spread fails. The mean *reported* standard error is fitted separately and labelled as what it is: an influence-curve SE is sigma-hat over root n, so its rate is near arithmetic and catches only a standard error carrying the wrong power of `n` |
| local efficiency and inference | at each size, the SE ratio is inside the sanity band, coverage clears the floor, and the bias interval is inside the equivalence margin | SE calibration is checked separately from bias, so correct centering cannot hide the wrong width, and it is measured at three sizes so a finite-sample gap is distinguishable from a wrong formula |
| type-I error | the 99% upper endpoint of the ATE rejection rate under a sharp null clears the nominal size by no more than the declared margin | the null law retains its confounding, so the test is not an unadjusted randomized comparison; and a separate cell under a real effect must reject, so a test that never fires cannot pass by being inert |

These statistical checks complement rather than replace the exact-law, Gateaux, remainder, and
identity instruments in the [estimand evidence manifest](../evidence.md#the-table).

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

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | replications per law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 34 | implementation-estimand tests against truth |
| `independent_tests_passed` | 34 | of those, passing |
| `paired_tests_total` | 17 | scenario-estimand paired tests |
| `paired_tests_passed` | 17 | of those, passing |
| `property_cells_total` | 13 | repeated-sampling property cells |
| `property_cells_passed` | 13 | of those, passing |
| `max_standardized_bias` | 0.133 | largest absolute bias in empirical standard deviations |
| `min_coverage` | 0.9356 | lowest measured coverage of a nominal 95% interval |
| `min_coverage_ci_lower` | 0.9182 | lowest exact 99% coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9057 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0728 | highest bootstrap SE-ratio endpoint |
| `max_se_ratio_resolution` | 0.0943 | farthest the widest SE-ratio interval reaches from 1.0 |
| `max_margin_utilization` | 0.6934 | largest share of a paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0121 | largest one-sided RMSE-ratio bound, against 1.10 |
| `min_coverage_difference_lower` | -0.0125 | smallest one-sided coverage-difference bound, against -0.025 |
| `max_calibration_excess_upper` | 0.0131 | largest SE-calibration-excess bound, against 0.05 |
| `properties[double_robustness/outcome_correct]:bias` | 0.000149 | bias with only the outcome nuisance correct |
| `properties[double_robustness/treatment_correct]:bias` | -0.01956 | bias with only the treatment nuisance correct |
| `properties[double_robustness/both_wrong]:bias` | -0.3348 | both-wrong negative control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5464 | fitted log-log sampling-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.6073 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4875 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.5056 | fitted reported-SE rate |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0325 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0627 | its 99% upper endpoint, against 0.10 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the positive control |
| `properties[crossfit_overfitting/stacked_cvtmle]:coverage` | 0.895 | coverage with cross-fitted tree predictions |
| `properties[crossfit_overfitting/stacked_cvtmle]:se_ratio` | 0.988 | SE calibration with cross-fitting |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.65 | coverage with the deliberately in-sample tree |
| `properties[crossfit_overfitting/in_sample_control]:se_ratio` | 0.5792 | SE calibration of that control |
| `properties[crossfit_overfitting/stacked_cvtmle]:coverage_gain_ci_lower` | 0.185 | paired 99% lower bound for coverage gained over the control |

### Comparison and property verdicts

Each implementation independently passes the ordinary study's 99% practical-bias, coverage, and
SE-calibration gates.  On the pairing, the 99% interval for the mean difference must lie inside
0.15 pooled empirical standard deviations.  Cleverly must also be non-inferior: the one-sided
RMSE-ratio upper bound is at most 1.10, the coverage-difference lower bound is at least -0.025, and
the excess absolute SE-calibration-error upper bound is at most 0.05 where the native inference
scales are comparable.  All 17 cells pass both the symmetric similarity and one-sided
non-inferiority claims, and both implementations remain independently valid.

The largest similarity-margin use occurs for continuous-law ATC, followed by ATT.  Both remain
inside the predeclared bound, and the RMSE, coverage, and calibration bounds remain well inside
their non-inferiority margins.  Continuous-law PAR has the study's lowest coverage, but its exact
99% lower endpoint still clears the 0.90 floor.  These are reported as the finite-sample cells that
use the most evidence budget, not tuned exceptions.

The 13 property cells use the same double-robustness, both-wrong discrimination, three-size
efficiency/calibration, empirical and reported root-n, confounded-null, and power instruments as
the fold-evaluated row below.  They add the same paired flexible-tree experiment: held-out
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

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | replications per law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 10 | estimand-law tests against truth |
| `independent_tests_passed` | 10 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 13 | repeated-sampling property cells |
| `property_cells_passed` | 13 | of those, passing |
| `max_standardized_bias` | 0.0455 | largest absolute bias in empirical standard deviations |
| `min_coverage` | 0.9413 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9244 | lowest exact 99% coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9338 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0668 | highest bootstrap SE-ratio endpoint |
| `properties[double_robustness/outcome_correct]:bias` | 0.000149 | bias with only the outcome nuisance correct |
| `properties[double_robustness/treatment_correct]:bias` | -0.01956 | bias with only the treatment nuisance correct |
| `properties[double_robustness/both_wrong]:bias` | -0.3348 | both-wrong negative control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5464 | fitted log-log sampling-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.6086 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4866 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.5059 | fitted reported-SE rate |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0325 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0627 | its 99% upper endpoint, against 0.10 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the positive control |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:coverage` | 0.895 | coverage with cross-fitted tree predictions |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:se_ratio` | 0.991 | SE calibration with cross-fitting |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.65 | coverage with the deliberately in-sample tree |
| `properties[crossfit_overfitting/in_sample_control]:se_ratio` | 0.5792 | SE calibration of that control |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:coverage_gain_ci_lower` | 0.1875 | paired 99% lower bound for coverage gained over the control |

### Statistical claims and controls

The first eleven property cells repeat the ordinary method's independent instruments: three
correct-nuisance combinations must have practically equivalent bias, both nuisances wrong must
be discriminated from that margin, calibration and efficiency must hold at three sample sizes,
the empirical-spread rate must be consistent with -1/2 and exclude -1/4, and a confounded sharp
null is paired with a nonzero-effect power control.  The reported-SE rate uses a predeclared
[-0.55, -0.45] practical root-n band and must exclude -1/4.  Requiring an increasingly precise
Monte Carlo interval to contain exactly -1/2 would eventually reject the arithmetic scaling for
a negligible finite-sample remainder.

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
