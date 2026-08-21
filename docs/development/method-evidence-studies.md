# R comparisons and statistical evidence studies

Use this guide when adding a method-level row to the
[evidence manifest](../evidence.md#method-evidence-grid). It is the reusable contract behind the
canonical TMLE studies, not a suggestion to invent a fresh validation protocol for each method.

## Start with the scientific claim

Write down the identified parameter, estimator construction, source paper, supported compositions,
and inference rule before choosing a reference implementation. An R package is a bounded
implementation witness: agreement can localize an engineering discrepancy, but it cannot establish
that the parameter, influence curve, or theorem was derived correctly.

Give different estimators different rows. Ordinary TMLE, stacked CV-TMLE, fold-evaluated CV-TMLE,
and a fold-specific-epsilon extension may share a limit while differing in finite samples. They may
not inherit one another's parity or coverage claims.

## Design the R comparison

1. Pin the R base image by digest and every package by an immutable commit or released source
   archive. Record those pins in the manifest and hash the Dockerfile and runner.
2. Give both implementations the same realized datasets, covariates, treatment contrast, nuisance
   model families, bounds, targeting controls, and interval scale. If a setting cannot be matched,
   declare the difference and exclude only the affected comparison.
3. For cross-fitted methods, give both implementations the exact same row-to-fold assignment. A
   shared fold count or random seed is insufficient because splitters and dependency versions can
   realize different partitions. Assert the assignment after constructing each implementation's
   fold object.
4. Preserve native inference scales. Compare log-scale RR/OR intervals on the log scale; do not
   force transformed intervals into a raw-SE comparison. Declare any exception on `StudyRecord`
   and add a test proving the scales really differ.
5. Abort on any failed or silently dropped replication. Never replace a failed sample, change its
   seed, or summarize a shorter run as the declared study.
6. Retain replication-level estimates, standard errors, interval endpoints, truth, coverage,
   pairing keys, and initial estimates where the reference exposes them. Raw simulated datasets may
   remain temporary.

Test each implementation against known truth before comparing them. The paired comparison uses
equivalence and non-inferiority margins declared before the run; it is not a test that the mean
difference is exactly zero. Require a nonzero targeting witness so agreement cannot be explained by
both implementations returning their initial plug-in estimates.

## Design the statistical study

Every property cell needs a law with an exact truth, a fixed estimator configuration, a replication
budget justified by the desired confidence interval, and a seed derived independently of the run's
size. Predeclare all margins before seeing the full results.

Two seed streams, with different rules. The *primary* replications come from the study's own record
through `replicate_seed`, so no two rows draw the same samples and a short probe redraws the head of
the published run. Property cells deliberately do *not*: their seeds are fixed constants shared
across studies, so every method is compared on identical draws, and a paired cell -- the cross-fit
arm against its in-sample control -- can merge on the replicate index. *Resampling* streams come
from `stream_seed`, which hashes a label rather than adding an offset. Adding offsets to a study
seed reads as separate and is not: consecutive study seeds and adjacent offsets hand two published
rows the same bootstrap index matrix, which makes intervals a reader compares side by side share
their Monte Carlo error.

Justify the budget against the binding gate rather than the one that reads tightest. A coverage
floor of 0.90 checked with an exact 99% interval needs 92.75% observed at 800 replications but
95.5% at 200 -- so at 200 a correctly calibrated 95% interval fails it more often than not, and the
cell discriminates nothing it did not already reject.

Positive cells need controls that make the same instrument fail:

- double robustness needs a both-wrong nuisance control;
- type-I error needs a nonzero-effect power control;
- coverage or SE-calibration claims need a deliberately invalid inference control;
- a convergence-rate claim needs at least three sample sizes and must exclude a materially slower
  rate, not merely contain the desired one. Accept it against a margin around the theoretical rate
  rather than by containment: containment is a point test that a bigger budget eventually fails,
  and half the distance to the slower rate you already declared is a margin you did not choose
  after seeing the answer.

Use interval-shaped verdicts. A bias equivalence interval should contract into a practical margin;
a coverage lower bound should clear a validity floor; and a one-sided type-I bound should rule out
material over-rejection. Avoid null-hypothesis tests of exact equality: with enough replications they
reject harmless finite-sample remainders and make stronger studies harder to pass.

Keep documentation examples out of the evidence count. Properties depending on repeated sampling,
large samples, or flexible learners belong in the named slow study and committed artifacts. Exact
laws, score identities, Gateaux derivatives, and deliberate mutations remain separate scientific
instruments and should be added whenever the method's construction makes them applicable.

## Register, regenerate, and publish

Add a `StudyRecord` with its scenarios, estimands, sample size, replication count, margins, runner
module, property module, expected property cells, artifact directory, document anchor, and every
module that determines the results. A study without an external comparator records a schema-valid
zero-row equivalence artifact and `0/0` paired tests; it must not invent a surrogate reference.

Regeneration must write these primary artifacts:

- `replicates.csv.gz` and `property-replicates.csv.gz`;
- `summary.csv`, `performance-tests.csv`, `equivalence.csv`, and `properties.csv`;
- `manifest.json`, including configuration, provenance, module hashes, reference hashes, and
  artifact hashes.

Run a disposable smoke study first. Then run the declared study without `--allow-failures`; a full
run that misses a margin is a finding, not permission to tune the margin after the fact. Quote
measured values only through `tests/studies/evidence/claims.py`, so the documentation gate checks
their rounding against the committed artifacts.

Finally run the targeted framework and documentation tests, the complete fast tier, and the named
slow method-evidence study. Do not run fast and slow tiers concurrently, and do not run the Python
and R full-core phases concurrently.

## Required limitations

State what the row does not reach: outcome and treatment types, missingness, observation weights,
clusters, fold repeats, learner class, truncation regime, interval type, simultaneous or bootstrap
inference, and any unsupported estimands. A composition absent from the samples is not implicitly
validated because its component parts have rows elsewhere.
