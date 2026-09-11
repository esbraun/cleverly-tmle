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

The beginner-facing computational path is `CausalStudy -> identify -> estimate`. `StudyProtocol`
owns the study context and assumption rationale. A design owns observed column roles. A typed
estimand owns the mathematical contrast and intervention metadata.

An `IdentifiedEffect` owns the observed-data functional and assumptions. A typed method owns
learning and runtime configuration.
The protocol can describe strategies in study language, but it cannot override another object's
contract. Do not reintroduce root engine constructors or a parallel string-driven convenience path.

One public question must normalize to one evidenced engine request. *Reconsider when* a distinct
audience has a workflow that cannot use these contracts without losing information. The
alternative must still converge to the same structured identification and result records.

An estimation method is named, never selected from the data. `estimate(method=...)` carries a
fixed default preset, which is a declaration rather than a choice. What is excluded is picking an
estimator by scanning `available_methods()`, or by comparing fits on the rows being estimated. The
temptation grows with the catalog. `riesz_tmle` and `ep` already appear there as
unavailable-with-reason, so a "use the best available method" convenience is one short function
away. It would report an interval whose selection step nothing certified and whose influence
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

Reported inference always belongs to the reported subset.
[Reporting a subset of a family](technical-reference/inference.md#reporting-a-subset-of-a-family)
states the rule and what is recomputed. The decision recorded here is that the public layer
recomputes rather than reusing the wider family's critical value. *Reconsider when* an engine can
be asked for the narrowed family directly, and the public layer stops selecting after the fact.

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

A supplied `SplitPlan` fixes outer validation assignments by input row position, and a plan read
off a result binds to that fit's data fingerprint. It does not fix inner learner or
collaborative-selection folds. Validate every repeat against that fingerprint before nuisance
fitting, including training support and cluster integrity. Results derive the public plan from
retained folds, and provenance fingerprints those same assignments. Targeted bootstrap,
longitudinal fits, and any refutation that changes the row set refuse a supplied plan until they
define their own row-mapping and validation contracts.
*Reconsider when* either engine records enough identity to validate a supplied plan before fitting.

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

A saved object round-trips the records it holds, and never a value it derived from them. Both
`TMLEResult` and each assessment facade drop every memoized value before joblib writes the artifact.
The shared filter reads the `cached_property` descriptors on the owning class, and it runs on both
sides of the pickle. A stored memo outlives the code that derived it, so the artifact reports a
conclusion this version does not reach. No cache generation invalidates a memo, because a memo
records no question. *Reconsider when* a derived value costs more to recompute than a load may
spend, and needs a stored generation instead.

Scalar result algebra is composed once in `inference.results`: sole-estimate selection, ordered
name validation, influence-curve extraction, joint covariance, and smooth delta-method contrasts.
Point and longitudinal result types delegate those operations and retain only method-specific
artifacts and reports. Scientific formulas that differ by method stay separate; identical result
algebra must not be copied into another result class.

Assessment is routed by declared fitted artifacts, not result-class names or parsed parameter
aliases. Every public result family has an explicit fit-wide supported, `deferred`,
`not_applicable`, or `unavailable` answer for every public diagnostic. A `deferred` capability
reports `available: False` and names the argument that makes the operation run. A cost opt-in stays
request state, because it changes no capability answer. Sensitivity selects parameters through
`ParameterKey`.

`validate()` summarizes only stored state and never refits, while refutation and benchmarking are
explicit expensive operations. Assessment caching is keyed by operation plus normalized arguments,
is persisted separately from estimates, and may not mutate the headline estimate or its summary.
A facade drops its memoized capability verdicts before it enters an artifact. The persistence
invariant above states that rule for every saved object.
*Reconsider when* an assessment needs stochastic state that cannot be normalized or serialized;
that operation must then declare itself non-deterministic from a saved result rather than entering
the persistent cache silently.

Both assessment facades route through one base. Lookup, refusal, and the combined report are
written once. A refusal therefore always carries the reason its own capability row declares, and a
combined report reads that declaration the same way on both facades. Sensitivity
implementations are reached through `SENSITIVITY_ROUTES`, which also declares whether the target
takes an estimand as its second positional argument; that table and the declared capabilities are
checked against each other in both directions. Whether an operation takes an estimand at all is
read from the routed signature, not from that flag, because an operation can take the same
ambiguous default by keyword. A facade may not fill in an estimand a fit leaves ambiguous:
substitution is for the case where exactly one reported parameter fits, and otherwise the analysis
refuses by name.

`run_all` sorts each included capability row into one of three execution classes. A direct alias
can remain explicit while its canonical row alone enters the combined report. Longitudinal
`stagewise` is such an alias for `support`.

A `summarize` row reads stored state and always runs. A `refit` row refits nuisances and runs only
under `include_refits`. A `retarget` row retargets cached nuisances and runs under
`include_retargets`, unless its own row declares `cost` as `cheap`. A cheap retarget runs by
default, which is how an eligible ordinary TMLE fit reports its derived E-value without refitting a
nuisance model.

The two flags stay separate because the two costs are disjoint. Refutation and benchmarking refit
nuisances. Point-treatment truncation curves usually retarget cached nuisances. Longitudinal
truncation curves refit the complete backward recursion while holding the mechanism predictions
fixed. One flag made whichever class it did not name run under the other's permission.

A row's execution class is a fact about the fitted result, not only about the operation.
`assessment_capabilities` resolves it per result. Guarded DR-TMLE refits the reduced regressions
inside its targeting alternation. Longitudinal TMLE refits every bound-dependent outcome and
pseudo-outcome regression and every targeting update. Both truncation curves therefore belong in
the `refit` class. A family still declares each operation exactly once, and the contract test
enforces that.

A longitudinal fit builds its truncation replay recipe before the first backward pass. The recipe
stores resolved plans, unfitted outcome and pseudo-outcome learner clones, and recursive solver
settings. It stores no fitted outcome model. The fitted result supplies the realized folds,
scaler, raw mechanism predictions, data, weights, clusters, and parameter structure. Replay
availability requires cloneable learners, and an explicit integer `random_state` wherever a learner
or a nested library member declares that parameter.
[Replay-only unavailability](technical-reference/scope-and-refusals.md#replay-only-unavailability)
states that rule and lists every omission code.

Every requested longitudinal grid starts with an exact fitted-bound replay. It compares all
retained estimates, regimen fits, and MSM fits, even when the grid omits the fitted pair.

`run_all` applies its gates in one order: caller deferral, then availability, then required
arguments, then cost. Every gate above the cost gate refuses for a reason no flag pays off. A
report that named the cost first told the caller to pass `include_refits=True` for a row that also
needs explicit `covariates`.

A combined run injects its top-level seed before those gates, not after them. A deferred row is a
request the caller can replay. A non-deterministic operation replays only with the seed the run
would have used. A row this fit refuses outright records no arguments, and that is the one
exception. It describes no invocation, so a seed on it names a draw that nothing ever took.

A missing required argument or cost opt-in is `deferred`, because the caller can make the operation
run. A missing method, derivation, replay artifact, or supported requested variant is `unavailable`.
An invoked operation that raises a capability refusal is also unavailable.

The deferral gate reads the capability status, not the supplied argument names. An argument that is
present with a refused value defers the same operation as an absent argument. The E-value defers on
`estimand=None`, which is its public default.

An ambiguous default estimand is a deferral on both facades. Several operations default to
`estimand="ate"` and answer for one parameter. A fit that reports several eligible parameters and
no bare `ate` leaves that choice to the caller. A fit that reports no eligible parameter stays
`unavailable`, because no argument makes a missing derivation run.

One eligible parameter is the case each facade answers for itself. A facade that substitutes that
name runs the row under it. A facade that substitutes nothing defers the row, because the
operation would otherwise run on the ambiguous default and refuse.

One predicate decides that ambiguity. Both the request-level capability resolution and the facade's
own parameter substitution read it. Written twice, the two disagreed: the substitution declined to
guess between two contrasts while the row beside it still advertised the analysis as runnable. The
combined report then invoked the operation and published the refusal as `unavailable`, under a next
step that named no argument.

Availability is authoritative before execution. Each capability row names the `Replayability` field
it needs in `requires_replay`, and the shared base applies that gate to every row. A facade may not
patch one row by name. `refute` read its slot only while running, and so reported `available=True`
on a result that carries no estimator.

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
measured `STUDY_JOBS` floor. It must still keep its phases *sequential*. `tests/canonical/tmle3/`
generates every sample and fits the Python side to completion before it hands the same samples to
the R container. The two are the same work on the same cores, so overlapping them would leave both
contending for a machine neither can have. The fast suite leaves inner parallelism alone.

Documentation examples are not statistical evidence. A unit, integration, or end-to-end test in
the fast tier must cover the behavior a guide shows, or a registered validation study must cover
it. A documented example never supplies the evidence for an estimate, an interval, or a diagnostic
verdict, whatever a runtime gate reads off that example. Evidence manifests such as
`docs/technical-reference/evidence.md` remain test-enforced source registries.

A reader-facing example must nonetheless *run*. `tests/unit/test_documentation_runtime.py`
executes the registered documents' fences. It runs two gates, and the table states what each one
covers.

| gate | documents | sample size | what it asserts |
| --- | --- | --- | --- |
| `test_every_example_runs` | every entry in `PRELUDES` | shrunk to `SMALL_N` | nothing raises |
| `test_tutorial_semantics_at_documented_size` | the four entries in `TUTORIAL_SEMANTIC_ASSERTIONS` | the size the page prints | one reviewed callback for that page |

The smoke gate asserts no number. It exists because compiling a fence cannot see a name the
package does not have. Six shipped examples were broken that way at once: two on a renamed
attribute, two calling `.summary()` on reports that expose `to_frame()`, one passing a float where
an assignment density is required, and one stratifying on a column the design does not adjust for.
Every one of them rendered as ordinary, copyable code. *Reconsider when* the check stops paying
for its runtime.

The semantic gate does assert numbers. It stays inside the rule above because of what it asserts
them about: each callback checks that a page reports its own output correctly. A callback reads
identification metadata and summary strings, exact display identities such as
`survival = 1 - risk`, and the direction, ordering, and coverage that the page narrates at the
page's own seed. An identity holds in every sample. A seeded relation holds in the one sample the
reader sees, so the callback detects a page that contradicts itself and certifies no method. A
method claim still needs an ordinary fast test or a registered study.

The gate is bounded by review rather than by a heuristic. Each callback lives in the test module
and not in the document, so a rewritten page cannot grant itself a numeric gate.
`test_every_tutorial_semantic_assertion_names_one_reviewed_runtime_example` pins the registry to
the four reviewed tutorials, and this second pass runs them unshrunk, which is the cost that keeps
the registry small. *Reconsider when* a callback needs a looser tolerance to keep passing. A
relation that moves under a supported change is a sampling claim, and it belongs in a registered
study.

A reader-facing notebook stores outputs, so its fast check is necessarily narrower. The stamp
detects a code, output, or stamp-field edit after stamping. It does not prove that the code produced
the output, because the hashes and payload share one mutable file. The executor disables skipped
cells and tolerated errors. A result-determining change requires its manual `--check` re-execution.

All other documentation checks are static and belong in the ordinary fast tier. Links resolve,
including links that name a repository path. Every `python` fence parses.

There is deliberately no manual documentation job. These properties are cheap enough to run on
every change, and a dispatch that re-ran them would read as a gate while adding no coverage. That
was the removed job's failure mode: its
`ruff check README.md docs` validated nothing at all, because the linter does not read Markdown.
Note the division: the ruff *formatter* does reach inside `python` fences and is covered by the
whole-tree `ruff format --check .`, but it skips any block it cannot parse, so syntax is a test's
job and not the formatter's.

**The warning-as-error build runs on every pull request, and that is a different job.** `ci.yml`'s
`docs` job builds the site with `-W` on Python 3.12, which is the interpreter `pages.yml` deploys
from. It is not the dispatch ruled out above, because it checks what no fast test reaches:
numpydoc validation of each rendered docstring, a document that no toctree references, and a
cross-reference Sphinx cannot resolve. Before it existed, `pages.yml` gave the first `-W` run
*after* the merge, so a rejected docstring took the published site down instead of failing a
request. A docstring must therefore validate on every supported interpreter and not on one:
`AssessmentStatus` documented the synthetic `enum.StrEnum` constructor that 3.12 exposes, which
left `PR01` and `PR02` firing on a 3.11 build of the same tree. *Reconsider when* the build stops
paying for its runtime on a request.

**Prose is reported, never gated.** `tests/prose.py` reports on reader-facing writing and
`tests/unit/test_documentation_prose.py` fails on a finding carrying no recorded judgment, not on
the writing. Recording `accepted: <reason>` in `tests/prose-report.md` is a passing outcome, so
the decision stays with the writer. *Reconsider when* a rule is found that is exact enough to
have no defensible exception. This is a standing decision rather than a preference, and the reason
is measured. A Vale rule that failed the build on an em dash produced a sweep that stripped dashes
mechanically, leaving six sentences without a predicate, two enumerations broken mid-list, five
altered technical claims and four deleted evidence clauses. The rule was correct and the
enforcement mode did the damage.

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
and maintenance cost.

Choose the algorithm before choosing the compiler. Newton targeting is the default because the
universal least-favourable one-step walk can dominate a cheap GLM fit, and the Gaussian multiplier
option avoids the Rademacher resampling matrix where its approximation is appropriate. Neither
choice removes the need to inspect overlap and influence-curve behavior.

Scale is constrained by statistical learning and memory before it is constrained by targeting
arithmetic. The conditional-density learner's long design is the remaining known superlinear
allocation. Benchmark with the intended learner and data shape before changing a numerical kernel.
