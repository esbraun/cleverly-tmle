# Architecture invariants

These are stable constraints that are easy to violate across module boundaries and are not fully
recoverable from any one implementation. The current architecture remains provisional, but a
change to one of these constraints must be intentional, documented, and backed by evidence.

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
add nested native threading without an end-to-end measurement and an oversubscription plan.

Concurrency is `outer × inner × threads-per-fit`; the third factor is pinned to one, and the split
of the first two belongs to the test tier. The fast tier consists of thousands of short tests, so
xdist balances it and inner `n_jobs` remains one. The docs tier is the mirror image: one test per
document and a long sequential namespace, so xdist has one useful worker and the budget goes
inward. That is sound only because `n_jobs` invariance is pinned bit for bit.

Size every layer from `tests/parallel.available_cores()`, which reads a container's CPU quota and
affinity mask through joblib. Neither `os.cpu_count()` nor xdist's `-n auto` does. Nesting pools is
sometimes right and is never assumed: record the configuration used and the measured alternative
where defaults are set.

Before adding compiled code, compare against a competent NumPy implementation and include the real
learner workload. Track compile time, memory, core count, numerical equivalence, and the kernel's
share of a fit. Generated benchmark output is not documentation; preserve environment metadata
and summarize durable conclusions in `docs/benchmarks/`.
