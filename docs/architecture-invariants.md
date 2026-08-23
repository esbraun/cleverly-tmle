# Architecture invariants

These are stable constraints and standing decisions that are easy to violate across module
boundaries and are not fully recoverable from any one implementation. The current architecture
remains provisional, and none of these is permanent: each holds until its stated **reconsider
when** condition is met, and a change to one must be intentional, documented, and backed by
evidence. When a condition is met, update the implementation, its independent evidence, and this
document in the same change. Superseded rationale belongs in Git history.

## Validation and evidence

Validate a derivation independently. Cross-language comparison against a canonical implementation
is a bounded secondary check and never the acceptance criterion: implementations descended from
one source share transcription errors, while derivative, exact-law, remainder, mutation, and score
checks fail against distinct error classes. The
[evidence manifest](technical-reference/evidence.md) records which instrument covers which
estimand, in both directions. The `LTMLE` fixture is the scoped exception,
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

A `K`-level arm enters a design matrix as `K - 1` drop-first indicators, through the shared
`data.validate.arm_indicators`. At `K = 2` that rule is the single 0/1 code column itself, which is
what delivers the bit-for-bit guarantee above rather than a separate compatibility branch. Both
`CausalData.treatment_block` and `LongitudinalData.history_design` call it, so a design that
conditions on an arm is coded the same way wherever it is built, and a longitudinal node's block is
sized by *that node's* level count rather than by a panel-wide one. One ordinal column is not an
acceptable simplification: it constrains any learner linear in its design to an ordered response in
the arm, which is a restriction on `Q` or on `g` that the estimand does not ask for. *Reconsider
when* an estimand is added whose treatment is genuinely ordered and whose derivation uses that
ordering. An ordinal coding would then be a modelling choice to declare, not a default to inherit.
Note that no exact-law test can see this choice, since a saturated learner partitions by distinct
design row and the two encodings are a bijection; the witness is
`tests/unit/test_sequential_design.py::TestAThreeLevelArmEntersAsIndicators`, a `glm` mechanism on a
non-monotone truth.

Internal tabular arithmetic stays in NumPy. The dataframe boundary is a negligible share of a fit,
supported learners consume NumPy arrays, and the public dataframe contract is already isolated
through narwhals, so a columnar engine has no share here to win. *Reconsider when* a supported
workload becomes dominated by joins, grouping, IO, or conversion rather than estimation.

## Public causal workflow

The beginner-facing computational path is `CausalStudy -> identify -> estimate`. A design owns
column roles, a typed estimand owns the causal question, an `IdentifiedEffect` owns the observed-data
functional and assumptions, and a typed method owns learning and runtime configuration. Do not
reintroduce root engine constructors or a parallel string-driven convenience path: one public
question must normalize to one evidenced engine request. *Reconsider when* a distinct audience has
a workflow that cannot be expressed through these contracts without losing information, and the
alternative still converges to the same structured identification and result records.

An estimation method is named, never selected from the data. `estimate(method=...)` carries a
fixed default preset, which is a declaration rather than a choice; what is excluded is picking an
estimator by scanning `available_methods()` or by comparing fits on the rows being estimated. The
temptation grows with the catalog. `riesz_tmle` and `ep` already appear there as
unavailable-with-reason, so a "use the best available method" convenience is one short function
away, and it would report an interval whose selection step nothing certified and whose influence
curve does not account for it. *Reconsider when* a published selector supplies its own influence
contribution and selection-aware inference, and certifies on draws that did not do the selecting.

Identification is complete before nuisance fitting. Unsupported estimand/design/provider/method
combinations fail at that boundary with a capability reason; a placeholder may not produce an
estimate for graph, Riesz, EP, front-door, IV, mediation, or transport behavior that has not been
implemented and evidenced. *Reconsider when* the corresponding work package supplies its
functional, method artifacts, persistence, and independent evidence.

A design and a prepared data container must agree on every role before either is used. Both
`CausalStudy` and the containers accept the same roles, and neither module alone can see a
disagreement: the container holds arrays that no longer name their source, while the design is
what `IdentifiedEffect.functional` records, `summary()` prints, and persistence writes. Adopting
a container without reconciling it therefore reports an adjustment set no estimate came from,
and saves it. *Reconsider when* a container carries its own identification record, so the design
has nothing left to contradict.

When the public layer reports a subset of the parameters an engine computed, the inference it
reports is the inference for that subset. A joint band is a statement about a family, so
narrowing the family and keeping the critical value asserts a coverage property over parameters
the result does not contain. Recompute from the retained influence curves under the same
significance level, draw count, multiplier distribution, seed, and cluster structure, so the
result is what the engine would have produced had it been asked for that family alone.
*Reconsider when* an engine can be asked for the narrowed family directly, and the public layer
stops selecting after the fact.

Where a configuration group serves more than one engine, a default that differs between them is
a sentinel resolved per engine, never a literal that silently picks one engine's answer for the
other. `g_bounds="auto"` and `n_multiplier="auto"` are the two current cases. A restated engine
default is pinned against that engine's signature by a test, because a restatement that nothing
compares is free to drift. *Reconsider when* the engines agree, at which point the sentinel
should be deleted rather than kept.

Configuration groups are immutable and normalized before engine construction. Convenience
keywords may map into `ModelSpec`, `CrossFitting`, `Targeting`, `Inference`, and `Runtime`, but may
not bypass them or reassign design roles. A shortcut whose name is a configuration field sets that
field; in particular `alpha` is the interval significance level and `submodel_alpha` is the
logistic-submodel bound.

A normalized method declaration either changes the selected engine request or fails before that
engine is constructed. Shared configuration groups do not imply shared implementation: every
non-default point-only setting is refused on a longitudinal design, while supported semantic
translations such as `cross_fit=False` to `n_folds=1` remain explicit. Method-configuration
failures derive from `CleverlyError`, so callers never have to catch implementation-language
`TypeError` or `ValueError` separately. *Reconsider when* an independently evidenced longitudinal
derivation makes a currently point-only setting operational.

Every causal result carries its `IdentifiedEffect`, normalized method, and structured
`ParameterKey` mapping. Persistence round-trips the complete result graph with joblib, including
the fitted arrays, estimator configuration, analysis data, and nuisance-estimator templates.
Loading therefore has pickle's arbitrary-code-execution risk and is restricted to trusted
artifacts in compatible dependency environments. *Reconsider when* a safe, estimator-agnostic
format can represent arbitrary third-party sklearn-compatible models without weakening replay.

Scalar result algebra is composed once in `inference.results`: sole-estimate selection, ordered
name validation, influence-curve extraction, joint covariance, and smooth delta-method contrasts.
Point and longitudinal result types delegate those operations and retain only method-specific
artifacts and reports. Scientific formulas that differ by method stay separate; identical result
algebra must not be copied into another result class.

Assessment is routed by declared fitted artifacts, not result-class names or parsed parameter
aliases. Every public result family has an explicit supported, `not_applicable`, or `unavailable`
answer for every public diagnostic; sensitivity selects parameters through `ParameterKey`.
`validate()` summarizes only stored state and never refits, while refutation and benchmarking are
explicit expensive operations. Assessment caching is keyed by operation plus normalized arguments,
is persisted separately from estimates, and may not mutate the headline estimate or its summary.
*Reconsider when* an assessment needs stochastic state that cannot be normalized or serialized;
that operation must then declare itself non-deterministic from a saved result rather than entering
the persistent cache silently.

Both assessment facades route through one base: lookup, refusal, and the combined report are
written once, so a refusal always carries the reason its own capability row declares and a
combined report reads the same declaration the same way on both. `run_all` names the two
expensive classes separately: `include_refits` for operations that refit nuisances and
`include_retargets` for those that retarget cached ones. They are disjoint, and one
flag made whichever class it did not name run under the other's permission. Sensitivity
implementations are reached through `SENSITIVITY_ROUTES`, which also declares whether the target
takes an estimand; that table and the declared capabilities are checked against each other in
both directions. A facade may not fill in an estimand a fit leaves ambiguous: substitution is for
the case where exactly one reported parameter fits, and otherwise the analysis refuses by name.

Repeated-sampling studies retain one structured `ReplicationRecord` per estimand and a
`ReplicationFailure` with replicate index, seed, exception type, and message for every failed
draw. Summaries are derived from those records through `summarize_replications`; evidence adapters
must not reimplement bias, root-n bias, coverage, or standard-error calibration. The study alpha
comes from the returned `ParameterEstimate`, and all successful records must agree on it.

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
can itself be a material cost.
*Reconsider when* a measured workload benefits from giving one model the machine; callers can
already opt out with `set_thread_limit(None)`.

Concurrency is `outer × inner × threads-per-fit`; the third factor is pinned to one, and the split
of the first two belongs to the test tier. The fast tier consists of thousands of short tests, so
xdist balances it and inner `n_jobs` remains one.

A standalone regeneration script is not a test tier and does not inherit that split. It owns the
machine, so it sizes its inner pool from `tests.parallel.available_cores()` rather than from the
measured `STUDY_JOBS` floor, but it must keep its phases *sequential*. `tests/canonical/tmle3/`
generates every sample and fits the Python side to completion before handing the same samples to
the R container, because the two are the same work on the same cores and overlapping them would
leave both contending for a machine neither can have. A slow-tier test that is the critical path of
its tier may take half the budget; the fast tier still leaves inner parallelism alone.

Documentation examples are not statistical evidence. Behavior shown in a guide must be covered by
a unit, integration, or end-to-end test in the fast tier, or by a named statistical study in the
slow tier, and no assertion about an estimate, an interval or a diagnostic verdict may rest on a
documented example. Evidence manifests such as `docs/technical-reference/evidence.md` remain
test-enforced source registries.

A reader-facing example must nonetheless *run*. `tests/unit/test_documentation_runtime.py`
executes the registered documents' fences and asserts only that nothing raises; it asserts nothing
about any number, which is what keeps the rule above intact. *Reconsider when* the check stops
paying for its runtime. It exists because compiling a fence cannot see a name the package does
not have, and five shipped examples were broken that way at once: two on an attribute that had been
renamed, two calling `.summary()` on reports that expose `to_frame()`, and one passing a float
where an assignment density is required. Every one of them rendered as ordinary, copyable code.

What is otherwise checked about the documentation is static, and belongs in the ordinary fast tier
rather than behind a dispatch. Links resolve, including links that name a repository path. Every
`python` fence parses. No reader-facing source joins two clauses with a dash, which Vale checks
for Markdown and `tests/unit/test_documentation_prose.py` checks for the notebook and the `.rst`
sources Vale cannot read. There is deliberately no manual documentation job. These
properties are cheap enough to run on every change, and a dispatch that re-ran them would read as
a gate while adding no coverage. That was the removed job's failure mode: its
`ruff check README.md docs` validated nothing at all, because the linter does not read Markdown.
Note the division: the ruff *formatter* does reach inside `python` fences and is covered by the
whole-tree `ruff format --check .`, but it skips any block it cannot parse, so syntax is a test's
job and not the formatter's.

Development-tool versions have one declaration: exact Ruff and mypy pins in the `dev` extra,
resolved by `uv.lock`. Nox and CI install that extra; they do not restate versions in session or
workflow files.

Size every layer from `tests/parallel.available_cores()`, which reads a container's CPU quota and
affinity mask through joblib. Neither `os.cpu_count()` nor xdist's `-n auto` does. Nesting pools is
sometimes right and is never assumed: record the configuration used and the measured alternative
where defaults are set.

Before adding compiled code, compare against a competent NumPy implementation and include the real
learner workload. Track compile time, memory, core count, numerical equivalence, and the kernel's
share of a fit. Machine-specific output is exploratory evidence, not durable documentation; only
the resulting cross-module decision and the condition that would reopen it belong here.

Production code stays pure Python. Prior measurements found nuisance fitting dominant in
representative workloads and found no material full-workload advantage from compiling the clearest
package-owned numerical kernels. *Reconsider when* a competent compiled implementation wins
materially in a full supported workload, including compilation, memory, data movement, packaging,
and maintenance cost. HAL is the clearest known workload likely to meet that condition.

Choose the algorithm before choosing the compiler. Newton targeting is the default because the
universal least-favourable one-step walk can dominate a cheap GLM fit, and the Gaussian multiplier
option avoids the Rademacher resampling matrix where its approximation is appropriate. Neither
choice removes the need to inspect overlap and influence-curve behavior.

Scale is constrained by statistical learning and memory before it is constrained by targeting
arithmetic. The conditional-density learner's long design is the remaining known superlinear
allocation. Benchmark with the intended learner and data shape before changing a numerical kernel.
