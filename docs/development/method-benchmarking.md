# Method benchmarking strategy

Method benchmarking has two complementary parts: bounded comparisons with an external R
implementation and independent statistical studies against properties implied by the derivation.
Neither replaces the package's exact-law, Gateaux, remainder, identity, and deliberate-mutation
checks. Use this strategy when adding or changing a method-level row in the
[implementation validation grid](../technical-reference/index.md#implementation-validation-grid).

## Begin with the scientific claim

Record the identified parameter, estimator construction, governing source, supported
compositions, and inference rule before choosing a comparator. An R package is an implementation
witness: agreement can localize a discrepancy, but it cannot prove that the parameter, influence
curve, or theorem is correct.

Give materially different estimators separate evidence rows. Ordinary TMLE, stacked CV-TMLE,
fold-evaluated CV-TMLE, and fold-specific extensions may share a limit while differing in finite
samples; one method does not inherit another's parity or coverage claims.

## R implementation comparisons

- Pin the R base image and packages immutably, and record hashes for the Dockerfile, runner, and
  reference sources in the study manifest.
- Give both implementations the same realized datasets, covariates, contrasts, nuisance-model
  families, bounds, targeting controls, and interval scale. Declare any setting that cannot be
  matched and exclude only the affected comparison.
- Supply and assert the exact row-to-fold assignment for cross-fitted methods. A common seed or
  fold count is not enough when splitters or dependency versions differ.
- Preserve native inference scales, including log-scale risk-ratio and odds-ratio intervals.
- Fail the run on a dropped or unsuccessful replication. Do not replace samples or summarize a
  shorter run than the one declared.
- Retain paired replication-level estimates, standard errors, interval endpoints, truth,
  coverage, initial estimates where exposed, and pairing keys. Require a nonzero targeting
  witness so parity cannot be explained by both implementations returning their initial fits.

Test each implementation against known truth before comparing them. Use predeclared equivalence
and non-inferiority margins for the paired comparison rather than a null test of exact equality.

## Independent statistical evidence

Each property cell declares an exact-truth law, fixed estimator configuration, replication budget,
margin, and seed rule before the full run. Size the budget against the binding confidence bound,
not the most favorable point estimate.

Primary replications use the study record's `replicate_seed`; property cells use fixed shared
seeds so methods see identical draws and paired controls merge by replication index. Resampling
uses independently labelled `stream_seed` streams rather than adjacent integer offsets.

Use interval-shaped verdicts: bias intervals must fit inside a practical equivalence margin,
coverage lower bounds must clear a validity floor, and one-sided type-I bounds must rule out
material over-rejection. Positive claims need a control that makes the same instrument fail:

- double robustness includes a both-wrong nuisance control;
- type-I error includes a nonzero-effect power control;
- coverage or standard-error calibration includes deliberately invalid inference; and
- convergence uses at least three sample sizes and excludes a predeclared slower rate.

Repeated-sampling, large-sample, and flexible-learner claims belong in the named slow study and
committed artifacts. Documentation examples never count as statistical evidence.

## Registration and acceptance

Register the scenarios, estimands, sample size, replication count, margins, runner and property
modules, expected cells, artifact directory, document anchor, and every result-determining module.
A study without an external comparator records a valid empty equivalence artifact rather than a
surrogate reference.

Published studies retain `replicates.csv.gz`, `property-replicates.csv.gz`, `summary.csv`,
`performance-tests.csv`, `equivalence.csv`, `properties.csv`, and a provenance- and hash-complete
`manifest.json`. Run a disposable smoke study first, then the declared study without permitting
failed replications or tuning margins after seeing the result. A regeneration expects the whole
machine: do not run the Python and R full-core phases concurrently, and do not run the fast and
slow test tiers concurrently. Documentation quotes measured values through
`tests/studies/evidence/claims.py` so tests can check them against the artifacts.

Every evidence row states what it does not cover, including relevant outcome and treatment types,
missingness, weights, clusters, fold repeats, learner class, truncation, interval type, and
unsupported estimands. Validate changes with the targeted evidence and documentation tests, the
complete fast tier, and only the named slow study whose path and assertion can observe the change.

## Adding a method row

A row is not written, it is earned by registering a study. The machinery in
`tests/studies/evidence/` is method-agnostic, so a new method supplies only what is genuinely its
own and inherits every gate, every negative control and the checks over the validation grid.
Follow the rest of this guide rather than designing new margins, fold-matching rules,
artifacts, or provenance conventions for each comparison.

1. **Declare it.** Add a `StudyRecord` to `tests/studies/evidence/registry.py`'s register: the
   name the validation grid's first cell must carry, the artefact directory, the document and
   section anchor, the scenarios and their estimands, the replication count and sample size, and a
   `Margins` contains every acceptance margin. Declare each margin before the run.
2. **Write only the method-specific half.** A law to sample from with an exact parameter oracle, a
   fit function and a reference implementation pinned by digest, when one exists. Seeds
   come from `replicate_seed` *applied to this study's own record*, so a replication is a fixed
   sample whatever the study's size and a short probe redraws the published one. Reusing another
   study's ready-made sampler inherits its seed silently while publishing your own, which a test
   now refuses: two registered studies may share a law but not a set of datasets.
3. **Say what a failure would look like.** Property cells are only evidence if a claim can fail:
   pair each positive cell with a control that must fail the same instrument in the opposite
   direction, and pair any rejection-rate cell with a power cell so an inert test cannot pass.
4. **Quote nothing by hand.** Numbers in the section go in its measured-values table by quantity
   name; the gate resolves each against the artefacts and checks it at the precision printed.
   The per-test tables are generated between sentinel comments by
   `python -m tests.studies.evidence.document`, so the section carries three empty
   `<!-- generated: ... -->` blocks and nothing typed into them.
5. **Name every key in plain English.** Add each new scenario, estimand, property family and
   cell to `tests/studies/evidence/descriptions.py`. A key the module does not describe fails
   the gate, so a study cannot publish a row whose meaning a reader cannot recover.
6. **Declare the blind spots.** The limitations cell is the column the grid exists for. A
   composition the study does not reach. Cross-fitting, weights, clustering, missing outcomes,
   and a flexible learner each require a separate row.

If a registered study has no canonical comparator, its cross-implementation cell says so and its
row rests on the other two columns; a study with no property cells is not a validation row at
all, because matching another implementation is not evidence that either one is right.
