# Architecture invariants

These are stable constraints and standing decisions that are easy to violate across module
boundaries and are not fully recoverable from any one implementation. The current architecture
remains provisional, and none of these is permanent: each holds until its stated **reconsider
when** condition is met, and a change to one must be intentional, documented, and backed by
evidence. When a condition is met, update the implementation, its independent evidence, and this
document in the same change. Superseded rationale belongs in Git history or in the underlying
evidence report.

## Validation and evidence

Validate a derivation independently. Cross-language comparison against a canonical implementation
is a bounded secondary check and never the acceptance criterion: implementations descended from
one source share transcription errors, while derivative, exact-law, remainder, mutation, and score
checks fail against distinct error classes. [`docs/evidence.md`](evidence.md) records which
instrument covers which estimand, in both directions. The `LTMLE` fixture is the scoped exception,
because it pins cumulative-bound placement and a nonzero finite-sample targeting path that exact
laws at `epsilon=0` cannot see. *Reconsider when* another named blind spot is demonstrated, the
compared implementations target the same estimand, and the comparison has predetermined pass/fail
actions.

Keep feature selection separate from statistical certification. Evaluating a configuration on the
draws that selected it makes the reported result selection-dependent, so a study that selects must
certify on draws that did not do the selecting. *Reconsider when* a study performs no
data-dependent selection, or uses disjoint selection and certification cohorts.

## Dataframes and labels

User-facing frames go through narwhals and results return in the caller's dataframe library. Keep
backend names, not whole input frames, in data and report objects so results remain serializable
and do not pin input memory. Cast numeric roles through narwhals before conversion; preserve the
treatment's logical type until treatment encoding. Detect null labels through the logical column
type, never by branching on pandas or polars.

Per-arm arrays are keyed by treatment level. Use shared arm-mapping helpers and carry original arm
labels into parameter names, tables, and errors. Do not parse user-visible names to recover arms;
compose the known names forward and retain a structured index.

The binary path is a bit-for-bit regression surface for generalized treatment support. A
multi-arm change must leave binary results unchanged unless a documented compatibility change is
intended.

Internal tabular arithmetic stays in NumPy. The dataframe boundary is a negligible share of a fit,
supported learners consume NumPy arrays, and the public dataframe contract is already isolated
through narwhals, so a columnar engine has no share here to win. *Reconsider when* a supported
workload becomes dominated by joins, grouping, IO, or conversion rather than estimation.

## Targets, interventions, and variants

A new point-treatment estimand is a `Target` registered through `targets.register`. If it needs a
new score group, register the fluctuation submodel first. Put its influence curve in the shared
inference layer so covariance, bands, delta methods, and score diagnostics remain reusable.

A regime is a density over arms and is a parameter axis distinct from arms, shifts, incremental
interventions, and MSM coefficients. Keep those axes explicit; a fit must not mix incompatible
definitions of its counterfactual under one result namespace.

An estimator variant that only changes which nuisance estimate is targeted should override the
nuisance hook, return a replaced `NuisanceEstimates` with diagnostics, and inherit targeting and
result behavior. `CTMLE` is the reference pattern. A method with different data ordering or
recursion, such as `LTMLE`, should use a separate container and result type and must explicitly
reuse or refuse each shared subsystem.

A selector-based `CTMLE` fit chooses one shared categorical treatment mechanism. At `K` arms its
selection target is one nonredundant vector: all `K` means for `ey`, or the `K - 1` contrasts
against the declared reference for `ate`, `rr`, and `or`. Do not fit or select a separate
mechanism per contrast under one result; that is a collection of estimators with different
nuisance states, not one joint fit.

`TMLE.retarget` operates on cached point-treatment nuisances. Do not assume this contract for a
variant whose derived equations require nuisance refits at the targeted state; document and test
the cost and persistence behavior instead.

## Longitudinal, survival, and competing risks

Resolve dynamic rules once into an assignment matrix before fitting. A rule receives only the
history available at its node, and downstream mechanism, follower, and outcome-regression logic
must read the same resolved plan. Outcome designs contain covariate history, not redundant past
treatment columns that are deterministic under the resolved plan.

Longitudinal MSMs are projections over regimen/horizon cells. Their fluctuation is pooled and the
backward recursion proceeds in lockstep over nodes; the horizon belongs in the design and each
cause receives its own projection while sharing nuisance fits.

For survival, a unit experiencing the event at node `t` belongs in node `t`'s event regression.
The event-free state is part of history, not an intervened mechanism factor. Reports store risk;
survival is a derived scale, with contrasts transformed by sign rather than by `1 - estimate`.

For competing risks, use a cause-specific event numerator with the all-cause survival factor.
Competing events affect the at-risk population but do not add a denominator factor unless a new
estimand explicitly intervenes on them. Cause-specific estimates need not be renormalized to a
simplex.

## Weights, bounds, and sensitivity

Observation weights define the target population and must flow through nuisance loss, score
equations, influence functions, covariance, and effective-sample-size calculations. Estimand or
MSM weights answer a different question; do not merge the two.

Choose mechanism bounds from the clever covariate's algebra. Conditional-effect groups use
propensity odds and need their corresponding bound regardless of arm count. A binary-only method
must declare and validate that restriction rather than indexing two arms by accident.

Sensitivity analyses operate on one named parameter or contrast. Resolve its arms from structured
parameter metadata, not string splitting, and preserve the distinction between assumptions shared
across arms and explicitly arm-specific assumptions.

## Parallelism and performance

Nuisance fits are single-threaded by default so parallelism occurs across folds and learner
candidates. Callers can change the process-level limit through the public learner controls. Do not
add nested native threading without an end-to-end measurement and an oversubscription plan. Nested
model parallelism oversubscribes small fits, and repeatedly constructing the thread-pool controller
was itself a major cost — see the [thread-limit profile](benchmarks/thread_limit_profile.md).
*Reconsider when* a measured workload benefits from giving one model the machine; callers can
already opt out with `set_thread_limit(None)`.

Concurrency is `outer × inner × threads-per-fit`; the third factor is pinned to one, and the split
of the first two belongs to the test tier. The fast tier consists of thousands of short tests, so
xdist balances it and inner `n_jobs` remains one. Pull-request documentation execution selects
only affected H2/H3 modules, each in a fresh namespace with its declared setup closure. The manual
full transcript remains one sequential namespace per document. Both documentation modes put the
budget inward, which is sound only because `n_jobs` invariance is pinned bit for bit.

Every executable documentation section and block has a stable ID. Section metadata names its
setup blocks and directly affected source paths; common or unclassified implementation changes
fail closed by selecting all ordinary sections. Statistical blocks never enter pull-request
execution. Syntax, links, catalogue members, and the metadata graph are checked over the complete
documentation set in the ordinary fast tier.

Put `doc-section: id=...; requires=...; paths=...` metadata immediately below each executable H2
or H3, and `doc-block: id=...; tier=fast|slow` immediately above each Python fence. Dependencies
name blocks, not whole sections, and may only name an earlier setup block in the same document.

Size every layer from `tests/parallel.available_cores()`, which reads a container's CPU quota and
affinity mask through joblib. Neither `os.cpu_count()` nor xdist's `-n auto` does. Nesting pools is
sometimes right and is never assumed: record the configuration used and the measured alternative
where defaults are set.

Before adding compiled code, compare against a competent NumPy implementation and include the real
learner workload. Track compile time, memory, core count, numerical equivalence, and the kernel's
share of a fit. Generated benchmark output is not documentation; preserve environment metadata
and summarize durable conclusions in `docs/benchmarks/`.

Production code stays pure Python and `numba` remains benchmark-only; nothing under `src/` imports
it. Nuisance fitting dominates representative workloads, and competent NumPy implementations
removed the apparent advantage in the clearest candidate kernels — the [benchmark
verdict](benchmarks/README.md#current-verdict) carries the evidence. *Reconsider when* a competent
compiled implementation wins materially in a full supported workload, including compilation,
memory, data movement, packaging, and maintenance cost. HAL is the clearest known workload likely
to meet that condition.

Choose the algorithm before choosing the compiler. Newton targeting is the default because the
universal least-favourable one-step walk can dominate a cheap GLM fit, and the Gaussian multiplier
option avoids the Rademacher resampling matrix where its approximation is appropriate. Neither
choice removes the need to inspect overlap and influence-curve behavior.

Scale is constrained by statistical learning and memory before it is constrained by targeting
arithmetic. The conditional-density learner's long design is the remaining known superlinear
allocation. Benchmark with the intended learner and data shape before changing a numerical kernel.
