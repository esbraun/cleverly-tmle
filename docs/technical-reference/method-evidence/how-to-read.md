# How to read these studies

Read this section once. It defines the terms and the rules that every study below applies.

## The three questions

A study asks three separate questions about `cleverly`. The counts are not interchangeable.

| question | what it establishes about `cleverly` | what it cannot establish |
| --- | --- | --- |
| **Accuracy against known truth** | `cleverly` clears declared bias and coverage-validity margins on a law whose truth is computed longhand | nothing about the derivation or exact nominal coverage. The canonical rows say nothing about `cleverly` |
| **Agreement with the canonical implementation** | `cleverly` and an independently maintained implementation compute the same thing on the same rows, within a declared margin | that either one is right. Two poor implementations can agree. This is why the accuracy question is asked first and separately |
| **Theory properties** | declared repeated-sampling properties, including robustness, root-n contraction, calibration, and error rates | behaviour outside the laws and compositions the cells declare. Efficiency requires a separate comparison with an independent bound |

The scientific derivation is checked elsewhere. The exact-law, Gateaux, remainder, identity, and
deliberate-mutation instruments are listed per estimand in the
[evidence manifest](../evidence.md#the-table). A study measures a complete estimator under repeated
sampling. It does not replace those instruments.

## The verdict rules

Every rule is an equivalence rule bounded by a margin declared before the run. None of them tests
whether a discrepancy is exactly zero, and the distinction is the design rather than a detail.

A Monte Carlo study accumulates evidence by adding replications. A verdict must therefore become
easier, or at worst stay put, as replications grow. A significance test does the reverse. It
converges on rejecting any estimator whose finite-sample remainder is not identically zero, which
is every estimator. The study would eventually go red for the one reason that is not a defect. An
earlier version of the first study carried two such rules. Quadrupling its replication count would
have failed it without a line of estimator code changing.
`tests/unit/test_evidence_framework.py` holds both rules side by side and asserts which way each
one moves.

| rule | what must hold | why it is bounded | its control |
| --- | --- | --- | --- |
| **bias equivalence** | the 99% Student interval for the error lies inside `margin:standardized_bias` empirical standard deviations of zero | a point test of zero bias fails every consistent estimator at enough replications | the same instrument read in the opposite direction: a control cell's interval must lie entirely *outside* the margin |
| **coverage validity** | the lower endpoint of the exact Clopper-Pearson interval clears `margin:coverage_floor` | whether a nominal 95% interval is valid is the question. Whether it is 95% to the third decimal is a question no finite study answers | one-sided by design. Over-coverage is conservative, not invalid |
| **SE sanity band** | the bootstrap interval for mean reported SE over empirical SD lies inside `margin:se_ratio_sanity_lower` to `margin:se_ratio_sanity_upper` | this is a screen for a standard error wrong by an order of magnitude. The coverage floor binds first | a ratio of 0.80 corresponds to about 88% coverage, so the band cannot be tightened past what the floor implies |
| **SE calibration** | where both nuisances are correct, the SE ratio interval lies inside the calibration band and the exact coverage interval inside its own | this is the only two-sided calibration claim, and the only gate a uniform tenth-scale understatement fails | a `shrunken_se_control` cell multiplies the standard errors by a declared factor and must fail |
| **efficiency** | the empirical standard deviation of the estimates and the mean reported standard error both sit inside `margin:efficiency_ratio_lower` to `margin:efficiency_ratio_upper` of an independently computed efficient-influence-function bound | Monte Carlo error in the empirical spread, which a point test against the bound would reject at enough replications. Calibration and validity are asked separately, because an estimator can have root-n bias and calibrated intervals without attaining the bound | a `noise_control` cell adds one bound-unit of independent noise and must fail |
| **paired similarity** | the 99% interval for the mean paired difference lies within `margin:paired_difference` pooled empirical standard deviations | symmetric, because a large difference in either direction means the two implementations are not computing the same thing | a paired mutation makes `cleverly` materially worse and must fail |
| **RMSE non-inferiority** | the bootstrap upper bound for `cleverly`'s RMSE ratio is at most `margin:rmse_noninferiority` | one-sided, because `cleverly` performing better than the reference is a result rather than a failure | a reference-only mutation must fail the reference's own column and leave `cleverly` standing |
| **coverage non-inferiority** | the lower bound for the coverage difference is at least `margin:coverage_noninferiority` | one-sided, for the same reason | as above |
| **coverage superiority** | the 99% lower bound for Cleverly minus reference coverage is positive, while Cleverly passes its truth gates and the RMSE and SE-calibration non-inferiority gates pass | superiority must improve a validity endpoint. A point-estimate difference or shorter runtime is descriptive | a comparison with equal coverage cannot pass this route |
| **calibration non-inferiority** | the upper bound for excess absolute SE-calibration error is at most `margin:calibration_noninferiority` | applied only where the two native inference scales are comparable | an exemption must be earned. A test requires the two implementations to report genuinely different scales |
| **type-I error** | the one-sided upper endpoint of the rejection rate under a confounded sharp null stays under `margin:type_i_ceiling` | the null law keeps its confounding, so the test is not an unadjusted comparison | a power cell under a real effect must reject. An inert test cannot pass by never firing |
| **power** | the rejection lower bound clears `margin:minimum_power` | this is the positive control the type-I cell needs | none. It is itself a control |
| **root-n rate** | the log-log slope interval lies within `margin:root_n_slope_lower` to `margin:root_n_slope_upper` | a band rather than containment of -1/2, which is a point test the reported-SE rate already fails at these replication counts | the interval must also exclude `margin:excluded_slope`, so a merely decreasing spread fails |

The **efficiency** rule needs a bound this package did not compute from the estimator it judges. A
study without one runs no efficiency cell, and its tables say so. The property family named
`root_n_and_efficiency` is the exception in name only. The name is historical, it tests bias,
coverage, and SE calibration across sample sizes, and the efficiency comparison lives in
`interval_calibration`.

Each implementation is judged on its own terms as well as on the pairing. A reference that degrades
is reported against the reference. It does not turn `cleverly`'s row red.

A study may declare particular estimands as a **point-only reference** when the external package
computes the point curve but does not report the influence curve required by this package's
parameter. Paired similarity and RMSE non-inferiority remain gated for those estimands. Reference
coverage and SE calibration remain visible in the accuracy table but are not compared, and the
manifest records both the estimands and the accepted reference failure. This declaration never
relaxes `cleverly`'s independent truth or property gates.

A paired row concludes `equivalent`, `superior`, `inferior`, or `inconclusive`. Equivalence and
superiority are separate passing routes. A reporting study can publish a failed scientific
verdict. It still refuses incomplete replications, invalid schemas, and broken provenance.
Failure to establish non-inferiority is `inconclusive`, not affirmative evidence of inferiority;
the `inferior` label is reserved for a future rule with a confidence bound in that direction.

The harness has negative controls for all of it, in `tests/unit/test_method_evidence.py`. The fast
tests corrupt bias, coverage, and reported standard errors for each implementation in turn. They
require only that implementation to fail and require the untouched one to keep passing.

## Reproducing a study

Each study page ends with a **Reproduction** section listing its own artifacts. Every study commits
the same set, and each file answers the same question wherever it appears.

| artifact | what it holds |
| --- | --- |
| the fixture `README.md` | the regeneration commands, in full and in smoke form |
| `manifest.json` | the configuration, every declared margin, every seed and package pin, the `cleverly` version and commit, the interpreter and library versions, and a hash of every published result and study module |
| `replicates.csv.gz` | one row per replication per implementation |
| `performance-tests.csv` | the accuracy verdicts against known truth |
| `equivalence.csv` | the paired verdicts against the canonical implementation |
| `properties.csv` | the theory-property cells and their controls |

A study page's Reproduction section adds only what is specific to that study.

## Terms

| term | definition |
| --- | --- |
| **replication** | one simulated dataset drawn from a study's law, fitted by every implementation. Seeds derive from the study's own record, so replication *k* is a fixed sample whatever the study's size |
| **law** | the data-generating process a scenario samples, with a parameter computed longhand rather than estimated |
| **estimand** | the parameter a row reports. Longitudinal keys carry their regimen in brackets |
| **cell** | one configuration of a property family, labelled `positive` when it must pass the family's instrument and `control` when it must fail it |
| **margin** | an acceptance threshold declared before the run and recorded in the study's `manifest.json`. Names beginning `margin:` in a measured-values table are these thresholds |
| **standardized bias** | mean error divided by the empirical standard deviation of the estimates. Scale-free, so it reads the same across estimands |
| **SE ratio** | mean reported standard error divided by the empirical standard deviation. One means the reported uncertainty matches the real spread |
| **share of margin used** | how much of a similarity margin a paired comparison consumed. Small means the two implementations agree far inside the bound |
| **plug-in** | the estimate the same nuisance fits produce with no targeting fluctuation. The distance from it measures what targeting did |
