# Method evidence studies

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
published run uses 400 replications of 1,000 observations for each of two laws:

- a bounded continuous-outcome law with effect modification, covering `ey1`, `ey0`, `ate`, `att`,
  `atc`, `ey_obs`, and `par`;
- a binary-outcome law covering those parameters plus `paf`, `rr`, and `or`.

The [per-replication results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/replicates.csv.gz)
contain the pairing key, truth, estimate, standard error, native interval endpoints, interval scale,
and coverage indicator. The [performance summary](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/summary.csv)
and [paired equivalence decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/equivalence.csv)
are recomputed from that file in the fast test suite. The separate
[independent performance tests](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/performance-tests.csv)
prevent agreement between two poor implementations from counting as evidence.

### Statistical tests at 99%

Each implementation is tested independently against known truth. A two-sided 99% Student interval
for inference-scale bias must lie within 0.25 empirical SD of zero. The exact two-sided 99%
Clopper--Pearson coverage interval must contain the nominal 95% rate and have a lower endpoint of at
least 88%. A deterministic 10,000-resample bootstrap 99% interval for mean reported SE divided by
empirical SD must lie within 0.80--1.20. These margins distinguish acceptable Monte Carlo
performance from numerical identity; they are recorded in the manifest and every result row.

Cross-implementation similarity uses the pairing. The two-sided 99% interval for the paired mean
difference must lie within plus or minus 0.15 pooled empirical SD. Because `cleverly` may perform
better than R, the remaining tests are one-sided 99% non-inferiority tests: the bootstrap upper
bound for its RMSE ratio must be at most 1.10, the bootstrap lower bound for its coverage difference
must be at least -0.025, and the bootstrap upper bound for excess absolute SE-calibration error must
be at most 0.05 when the SE scales are comparable. A failure is an implementation investigation and
potential `cleverly` fix; better performance is accepted.

The harness has method-specific negative controls. Its fast tests independently corrupt bias,
coverage, and reported SEs for each implementation, require only that implementation to fail, and
require the untouched implementation to keep passing. A separate paired mutation makes
`cleverly` materially worse and must fail both 99% similarity and non-inferiority. Thus the result
cannot pass merely because both methods share a dataset, nuisance specification, or summary code.

### Measured results

All 34 independent method-estimand tests pass: 17/17 for `cleverly` and 17/17 for R `tmle3`. Across
them, the lowest exact 99% coverage-interval endpoint is 88.75%, and the bootstrap 99% SE-ratio
intervals span no farther than 0.851--1.182. Every bias interval is contained in its standardized
equivalence margin.

All 17 paired similarity and `cleverly` non-inferiority tests also pass. The closest paired 99%
interval uses 79.5% of its equivalence margin. The largest one-sided 99% RMSE-ratio upper bound is
1.0163, the smallest coverage-difference lower bound is -0.0200, and the largest comparable
SE-calibration-excess upper bound is 0.0146. Thus the 99% tests establish that `cleverly` is at least
as good as the R reference within the declared practical margins; they do not penalize it when it
performs better.

Continuous-outcome ATC retains a nonzero paired mean difference of 0.000795, but its 99% interval
is contained in the declared 0.15-SD similarity margin. Both implementations have identical 92.75%
coverage and differ in RMSE by 0.15%. This study therefore supports statistically established
performance similarity, not estimate-by-estimate numerical identity. Verify every value in the
[summary](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/summary.csv)
and [equivalence table](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/equivalence.csv).

That conditional-effect discrepancy was evaluated as a possible `cleverly` defect. It does not
currently justify a production fix: for continuous ATT and ATC, `cleverly` has smaller absolute
bias than the constrained R path (0.000134 versus 0.000832 for ATT; 0.000044 versus 0.000751 for
ATC), with essentially unchanged RMSE and coverage. The gate remains asymmetric so a future
change that reverses this conclusion will fail the evidence suite rather than be documented away.

### Confidence intervals by estimand

| estimands | cleverly interval | R `tmle3` interval | comparison |
| --- | --- | --- | --- |
| levels, ATE, ATT, ATC, PAR | identity-scale Wald | identity-scale Wald | estimate, SE, endpoints, and coverage |
| RR and OR | log-scale Wald, exponentiated | log-scale Wald, exponentiated | estimate, SE, endpoints, and coverage |
| PAF | identity-scale Wald from the PAF influence curve | negative-log-complement (log-risk-ratio) Wald transformed by `1 - exp(-x)` | point performance and coverage; raw SE parity is not claimed |

That PAF distinction is not papered over. The two intervals have the same first-order delta-method
limit but need not have the same finite-sample endpoints, so forcing their raw standard errors into
one parity threshold would compare different reported scales.

### Properties checked independently

The [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/properties.csv)
and their [replication-level data](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/property-replicates.csv.gz)
record four nuisance-model cells, two sample sizes, and a sharp-null experiment:

| claim | positive evidence | negative control or discriminator |
| --- | --- | --- |
| double-robust consistency | both nuisances correct, outcome regression only correct, and treatment mechanism only correct have bias within Monte Carlo resolution | both nuisances wrong must have bias above four Monte Carlo SEs and 0.10 |
| root-n behavior | quadrupling sample size approximately halves the mean reported SE and does not make root-n bias diverge | the two sizes are evaluated separately rather than inferring a rate from one sample |
| local efficiency and inference | empirical sampling SD agrees with the mean influence-curve SE and coverage remains at least 0.90 under correct parametric nuisances | SE calibration is checked separately from bias, so correct centering cannot hide the wrong width |
| type-I error | ATE rejection under a sharp null is within three binomial Monte Carlo SEs of 0.05 | the null law retains confounding, so the test is not an unadjusted randomized comparison |

These statistical checks complement rather than replace the exact-law, Gateaux, remainder, and
identity instruments in the [estimand evidence manifest](../evidence.md#the-table).

The measured double-robustness biases are 0.007 with both nuisances correct, -0.013 with only the
outcome regression correct, and -0.023 with only the treatment mechanism correct; the both-wrong
negative control has bias -0.340 and only 20% coverage. Quadrupling sample size from 500 to 2,000
reduces the mean SE from 0.0905 to 0.0456, while coverage is 95.5% at both sizes and SE ratios are
0.943 and 1.026. Under the confounded sharp null, rejection is 3.25% and coverage is 96.75%.

### Reproduction and provenance

The complete regeneration instructions and the PAF qualification are in the
[`tmle3` fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/README.md).
The container pins R 4.5.2 by digest, `tmle3` at `ed72f8a`, and `sl3` at
[`0e8f236`](https://github.com/tlverse/sl3/tree/0e8f2365bcbe54010b8120c04a7a2dcfc8119227).
[`manifest.json`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/manifest.json)
records the study configuration and hashes every published result.

The public R specifications are used directly except for ATT and ATC, which use the constrained,
one-dimensional updater exercised in the pinned package's own tests. The public ATT convenience
path fails to converge on 5/400 bounded-continuous samples; the package-tested path is fixed for
all replications, and no failed sample is discarded or replaced.

The study does not establish parity for cleverly's default cross-fitting, repeated cross-fitting,
CV-TMLE evaluation, simultaneous intervals, bootstrap inference, missing outcomes, weights,
clustering, strata, multi-valued treatment, or flexible learners. Each is a distinct method or
composition and needs its own evidence row.
