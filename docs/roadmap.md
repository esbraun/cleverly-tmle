# Roadmap

What ships today, where its limits are, and what is worth building next. This is a
forward-looking document: detailed records of completed investigations belong in Git history
and release tags, while current contracts and evidence stay on `main`.

## Current status

`cleverly` is an alpha-stage library with a broad implemented surface, not a feature-complete
one. The [user guide](user-guide.md) is the capability reference and the [technical
appendix](methodology.md) gives the corresponding estimands, influence functions, assumptions,
and validation strategy.

`cleverly.DRTMLE` is **released under conditional validity**. Its production contract is
[DR-TMLE: doubly-robust inference](drtmle.md): the implementation agrees with the source
derivation, its component identities and score equations are regression-tested, and its interval
is valid conditional on adequate primary and reduced-regression nuisance fits. A fit cannot verify
those statistical rate conditions from its own output.

The completed DR-TMLE validation programme is archived at the
`drtmle-validation-archive-2026-08` tag. That archive contains the study harnesses, replicate
records, differential diagnostics, dispatch workflows, and working notes. Two issues that once
blocked release—the mechanism-correction sign review and centring under bounded mechanism
targeting—are closed and covered by theorem, derivative, score, and regression tests on `main`.

## Variants

Estimator variants share the package's data, learner, targeting, inference, sensitivity, and
validation infrastructure. New variants should reuse those layers and document only the behavior
they change.

- **Point-treatment TMLE (`cleverly.TMLE`)** supports binary and multi-valued treatments,
  deterministic and stochastic interventions, shifts, incremental interventions, marginal
  structural models, missing outcomes, controlled direct effects, weights, cross-fitting, and
  repeated cross-fitting.
- **Collaborative TMLE (`cleverly.CTMLE`)** provides greedy, ordered, and discrete propensity
  selection with one shared categorical path for any discrete treatment, plus ctmle3-style
  outcome-adaptive treatment modelling through `strategy="oat"`.
- **Doubly-robust inference (`cleverly.DRTMLE`)** is released for its documented point-treatment
  scope, including binary randomized trials with MAR outcomes under the published no-cross-fitting
  construction. It estimates the reduced regressions and additional corrections needed for an
  interval that remains valid when one primary nuisance is inconsistent, subject to the conditions
  in its [contract](drtmle.md).
- **Longitudinal TMLE (`cleverly.longitudinal.LTMLE`)** is released for static and dynamic
  regimens, time-varying confounding, monotone censoring, survival outcomes, competing risks,
  observation weights, and working marginal structural models over regimens.

## What is still open

### Current limitations

These are properties of the current methods or implementation, not live release defects.

- **DR-TMLE inference remains conditional on nuisance quality.** Numerical score convergence does
  not establish the required primary or reduced-regression rates. The archived finite-sample
  studies showed a material improvement over ordinary TMLE in the intended one-bad-nuisance
  regime, but did not establish nominal coverage at the reachable sample sizes or isolate the
  remaining finite-sample remainder.
- **DR-TMLE's targeting alternation has no general convergence proof.** Every returned fit exposes
  convergence, conditioning, and score diagnostics; callers must inspect them. The mechanism
  equation is solved iteratively because its covariate depends on the mechanism being tilted, so
  its final residual is small rather than algebraically zero.
- **DR-TMLE retargeting may require nuisance refits.** The reduced regressions are defined at the
  targeted state. Consequently a truncation or missingness sweep can cost about as much as a fit,
  and a deserialized result cannot perform an operation that requires learners which were not
  serialized.
- **Some sensitivity paths are intentionally fixed at fit time.** In particular, the reduced
  `gr2` regression contains the fitted mechanism in its target and cannot be losslessly
  reconstructed by re-truncating a stored array. Diagnostics identify this behavior rather than
  presenting a flat curve as evidence of insensitivity.
- **Several DR-TMLE compositions still lack published support.** Cross-validated DR-TMLE needs its
  published fold construction reconciled with this package's targeting and evaluation semantics;
  bivariate reduction still needs van der Laan (2014), Theorem 3 read into the contract. Missing
  treatment remains refused: canonical smoke tests are implementation provenance, but no located
  theorem supplies the identified parameter, corrected curve, and remainder. The future public
  API name `treatment_delta=` is reserved rather than accepted.
- **Scale is constrained by statistical learning and memory before targeting arithmetic.** The
  conditional-density learner's long design is the remaining known superlinear allocation.
  Benchmark with the intended learner and data shape before changing numerical kernels.

### Candidate features

The following are well-posed gaps, not promises or defects. A contribution should begin with the
derivation in the technical appendix and the acceptance instruments in `docs/evidence.md`.

- cross-validated DR-TMLE, followed by bivariate reductions and the remaining published DR-TMLE
  compositions;
- an MNAR tilt whose estimand is derived for continuous-dose shifts;
- intermediate variables with incremental interventions;
- multi-valued nodes, targeted bootstrap, persistence, and sensitivity analysis for LTMLE;
- blocked-temporal and rolling-origin cross-fitting;
- replicate-weight designs such as BRR and jackknife;
- additional longitudinal estimands, including interventions on competing events, once their
  separate identification assumptions and influence functions are specified;
- HAL and undersmoothed HAL learners, with the native implementation needed for their basis and
  optimization workload.

The full refusal taxonomy remains in [How to read a refusal](methodology.md#how-to-read-a-refusal).
It distinguishes missing package functionality from a different causal question and from a method
that would be wrong by construction.

## Standing decisions

These are current engineering decisions backed by evidence. They are constraints on an
implementation until their reopening condition is met, not declarations that the design can
never evolve.

| decision | current evidence | reopen when |
| --- | --- | --- |
| Keep production code pure Python; keep `numba` benchmark-only | nuisance fitting dominates realistic workloads, and properly written numpy removed the apparent wins in the clearest candidate kernels | a competent compiled implementation wins materially in a full supported workload, including compile and data-movement cost |
| Keep internal tabular arithmetic in numpy | the dataframe boundary is a negligible share of a fit and supported learners consume numpy arrays | a supported workload becomes dominated by joins, grouping, IO, or conversion rather than estimation |
| Parallelize across folds and learner candidates; run individual nuisance fits single-threaded by default | nested model parallelism oversubscribes small fits, and constructing the thread-pool controller repeatedly was itself a major cost | a measured workload benefits from giving one model the machine; callers can already opt out with `set_thread_limit(None)` |
| Validate derivations independently; use cross-language comparison only as a bounded secondary check | implementations descended from the same source share transcription errors, while derivative, exact-law, remainder, mutation, and score checks fail against distinct error classes. The `LTMLE` fixture is the scoped exception: it pins cumulative-bound placement and the nonzero finite-sample targeting path, which exact laws at `epsilon=0` cannot see | another named blind spot is demonstrated, the compared implementations target the same estimand, and the comparison has predetermined pass/fail actions |
| Keep generated benchmark results out of Git | timings are properties of the recorded hardware and environment, not timeless package facts | never for unlabelled raw results; durable conclusions belong in reviewed reports with reproducible commands and environment metadata |
| Separate feature selection from statistical certification | evaluating a configuration on the draws that selected it makes the result selection-dependent | a study performs no data-dependent selection, or uses disjoint selection and certification cohorts |

## Native acceleration

The current benchmarks support numpy for production kernels. Nuisance estimation dominates a
realistic fit and already executes in compiled third-party code; cleverly-authored targeting and
dataframe work is generally a small share. Past apparent compiler wins shrank after improving the
numpy baseline, removing redundant encoding, or measuring the kernel as part of a full fit.

This is not a permanent ban. Revisit native code when one of these conditions holds:

- HAL or another supported learner moves substantial estimation work into package-owned basis
  construction and optimization;
- profiling a real supported workflow shows package-owned arithmetic dominates end-to-end time;
- a compiled kernel beats the competent numpy implementation on representative sizes and core
  counts without unacceptable compile, memory, packaging, or maintenance cost.

Use `benchmarks/bench_tmle.py` for end-to-end shares and `benchmarks/numba/` for isolated kernels
and post-nuisance pipelines. The [benchmark guide](benchmarks/README.md) records the current
verdict, focused evidence, controls, and commands. Always include a realistic learner preset:
`library="glm"` intentionally makes every non-learner line look larger than it does in the default
library.

At very large `n`, choose the algorithm before choosing a compiler. Newton targeting is the
default because the universal least-favourable one-step walk can dominate a cheap GLM fit, and the
Gaussian multiplier option avoids the Rademacher resampling matrix when its approximation is
appropriate. Neither choice changes the need to inspect overlap and influence-curve behavior.
