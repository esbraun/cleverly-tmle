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
away. It would report an interval whose post-selection coverage nothing certified and for which no
selection contribution, common-influence-curve reduction, or stable-oracle reduction had been
derived. *Reconsider when* a published selector supplies selection-aware inference and certifies
on draws that did not do the selecting.

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

A point-treatment estimator stamps the inference status in `TMLE._retarget_detailed`, from its
`_inference_status(data)` hook. The hook reads the estimator configuration and the prepared
`CausalData`, and nothing fitted. The status can be determined before learner runs, though
`_retarget_detailed` stamps estimates after nuisance fitting.
`fit`, `retarget`, and each sensitivity sweep get their estimates from that method, so no sweep
builds an interval that the fit refuses. The same method stamps both fold-level reports, and
`CVTargeting.inference` reads the status from those reports rather than store it.

The longitudinal estimator decides its fit status in `_inference_status(data, folds)`, in
`cleverly.longitudinal.estimator`. That function calls `cluster_inference_status` on the prepared
cluster labels and weights, and it reads nothing fitted. `LTMLE.fit` and the truncation-curve
replay `_refit_bound` pass the fit status to `_estimates` and `_msm_estimates`. Those two builders
stamp it on each estimate. The replay must equal the fit field for field, so a stamp in `fit`
alone would make `truncation_curve()` refuse.

A parameter status sits beside the fit status. `PARAMETER_STATUSES` in
`src/cleverly/_inference_status.py` names the statuses that one parameter takes, and today it
holds `constant_node_plugin` only. `_estimates` stamps it through `_parameter_status`, in the fit
and in `_refit_bound` alike. `_parameter_status` reads one fitted quantity, `constant_nodes(fit)`:
the nodes whose regression read a constant 0 or 1 over the whole sample. A level takes the status
when its own fit has such a node, and a contrast when its fit or its reference fit has one. A fit
status that withholds inference comes first in `NON_INFERENTIAL`, so it stays.

| path | status it sets |
| --- | --- |
| `smooth_contrast` and `median_estimates` | the status of their inputs, through `inference_status`. A mix of two fit statuses raises `ValueError`. A mix of a fit status and a parameter status takes the earlier one in `NON_INFERENTIAL` |
| `variable_importance` | none. It raises the fold-policy refusal first. On each candidate's prepared data it raises the outcome-scale refusal, then asks the hook. It refuses before the first fit. A cross-fitted run of a continuous outcome with no `q_bounds` meets the scale refusal first in two cases. With `delta=`, its fit would raise the arm-indexed refusal first. With a `CTMLE` template, the hook would give its collaborative status |
| a longitudinal estimator | `cluster_inference_status` on the prepared cluster labels and weights. The fit and the truncation-curve replay pass it through `_estimates` and `_msm_estimates` to each `make_estimate` call. The replay computes the status again from the data and folds of the result, so the replay at the fitted bound equals the fit in every field that `_fitted_replay_matches` compares |
| a longitudinal parameter | `_parameter_status` in `_estimates`: `constant_node_plugin` when `constant_nodes` of its fit, or of its reference fit for a contrast, is not empty. `_msm_estimates` never meets it, because `msm=` refuses a constant cell in the backward pass |
| `incidence_total()` | each total's own status, through `inference_status` over the incidences it sums |
| a band | none. `LTMLE._bands` and `CausalStudy`'s `_narrow_bands` build the band over the estimates that supply inference, and below two of them they build none |

The same stamp sets the Student $t$ reference of a clustered fit. A builder records the rows an
estimate reads, and not its degrees of freedom: `_stratum_estimates` returns the stratum code of
each stratum estimate. `stamp_inference` turns that record into `reference_df`, and it does so
only when the fit's status supplies inference. A fit at `"few_cluster_plugin"` therefore keeps
`reference_df=None` on every estimate, and no stratum count below the floor reaches
`cluster_reference_df`.

A derived estimate takes the smallest `reference_df` of its inputs at its own `make_estimate`
call. Every interval, p-value and sensitivity limit reads the reference
through `wald_ci`, `wald_statistic`, `reference_quantile` or `one_sided_quantile` in
`cleverly.inference.delta`, and `tests/unit/test_few_cluster_reference.py` scans the package for
a second normal quantile. A future fold-evaluated report with baseline strata must carry each
stratum's row set into `_average_over_folds`, or its stratum estimates read the fit's cluster count.
*Reconsider when* a source gives a reference that is not a function of the cluster count.

One fit has one fit status. When more than one non-inferential fit status applies, the fit takes
the first one in `NON_INFERENTIAL` (`src/cleverly/_inference_status.py`). An override that finds
more than one status passes them to `precedent_status`, so the order lives in the table. A
parameter status can sit beside it on some estimates. The result's `inference_status`, through
`reported_status`, is the fit status: it leaves parameter statuses out unless every estimate holds
one.

A reader of `std_error`, `ci`, or `pvalue` must branch on each estimate's `supplies_inference`.
The `inference_status` of a result is the fit status, so it alone does not say that every
estimate supplies inference. A reader that decides columns from it must also handle the estimates
that do not: `summary()` stars them, `curve()` and `incidence_total()` add an `inference` column
and the plug-in columns, and `to_frame()` holds the union of the columns.

A frame or label that
publishes a spread must take its names from `spread_columns()` or `spread_name`, which raises
`KeyError` for an inferential name with no diagnostic entry. A text that names a status must read
it from `NON_INFERENTIAL`. The estimator that stamps a parameter status names it through
`CONSTANT_NODE_STATUS`. [Inference status](technical-reference/inference.md#inference-status)
gives the public contract. *Reconsider when* [F18](roadmap.md#f18-selector-path-c-tmle-inference)
supplies the selector paths' influence curve, when a second parameter status is added, or when
[F31](roadmap.md#f31-inference-at-a-boundary-node-estimate) supplies an interval at a boundary
node, which would remove the one status that reads a fitted quantity.

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
fitting, including cluster integrity. Results derive the public plan from
retained folds, and provenance fingerprints those same assignments. The full-refit bootstrap,
longitudinal fits, and any refutation that changes the row set refuse a supplied plan until they
define their own row-mapping and validation contracts.

A fit accepts a supplied plan only when the plan records how `random_partition` drew each repeat.
Draw each repeat again from that record before nuisance fitting, and refuse labels the record does
not produce. Labels no recorded draw produces could have been chosen by reading the outcome, which
is the leak cross-fitting prevents. A result carries the record of every retained draw, and a
refit keeps the record and drops the fingerprint binding. `SplitPlan` stays constructible from
labels alone, and every layer that accepts a plan refuses one with no record.
*Reconsider when* another generator records enough identity to be drawn again from its record.

An outer split reads the row count, the cluster labels and a seed, and it reads no treatment,
outcome or covariate value. No fit draws a partition that balances the data it then conditions on.
The declaration layer refuses a balancing policy before any split exists, and the fit layer refuses
it again, because a copied or modified estimator can carry a policy that the package does not draw
under. A selector-based collaborative fit draws selection folds without cross-fitting, so the
refusal reaches it at every setting. Outcome-adaptive C-TMLE draws no selection folds, so its
in-sample fit accepts an unused policy. *Reconsider when* a reviewed result covers a partition read off the analysed
data for a shipped estimator.

A cross-fitted fit of a continuous outcome works on a scale the caller declared, and never on one
the sample supplied. Refuse `q_bounds=None` before nuisance fitting, in every engine that
cross-fits. No package or study code derives an outcome support from a realized sample. The package
checks only that a declared interval contains the observed outcomes. *Reconsider when* a reviewed
result covers a cross-fitted transform whose endpoints the held-out rows helped choose.

What stratification used to guarantee is now checked on the realized draw, before the first
learner. Each treatment arm, and each class of a binary outcome, must reach two independent units.
Each training complement must carry every arm, a row whose outcome was observed, and every class a
learner's inner split needs. The independent unit is the row, and
it is the cluster where `id=` declares one. A refusal raised after a draw names no redraw, no seed
and no fold count, because a split that happens to fit was chosen by reading the values the draw
must not read. *Reconsider when* a split law that reads the treatment acquires a reviewed result.

A composition the package cannot support is refused by name before it costs a learner. Collaborative
TMLE refuses declared clusters at every setting, because no reviewed result covers a grouped draw
of its selection splits. *Reconsider when* a cluster-level result covers the selection path. Every
outer split of a clustered fit, point treatment or longitudinal, is whole-cluster. `make_folds`
checks it for point treatment, and `LTMLE.fit` checks it with `check_integrity` after the draw.

An MSM projection weight, an MSM design, and a `Stochastic` density are user-supplied functions
that the reported influence curve treats as fixed. Each carries a declaration with `"known"`,
`"estimated"`, or `None`, and only `"known"` fits. A callable can close over an estimate, and code
cannot inspect that closure. `MSM.linear` leaves `design_kind` as `None`, because the design it
builds is known by construction. An MSM with no design declaration reads as `"known"` only when its
design has the exact type that `MSM.linear` builds.

The MSM and `Stochastic` objects refuse undeclared and estimated functions at construction. Their
estimator paths check again before a fit or a result recomputation, because a copied or modified
object can carry an invalid declaration. The `TMLE` fit checks before each refusal of the fit
configuration, because no remedy that those refusals name lets an undeclared function fit. The
public evaluators `MSMSet.evaluate`, `evaluate_regimen_msm`, `RegimeSet.evaluate`, and
`Stochastic.density` check before they run a user function. The simulated-confounding replay
evaluates through them. Its replay model reads each MSM declaration from the source model, under the
same rules as the fit.

A treatment rule and the density of a user-written `Intervention` carry the same three-state
declaration. The table gives where each one refuses `None` and `"estimated"`. Each refusal comes
before any learner and before the function runs.

| function | declaration | refused at | checked again by |
| --- | --- | --- | --- |
| `Rule.rule` | `rule_kind` | construction, and the `TMLE` fit | each retarget, `Rule.density`, and `RegimeSet.evaluate` |
| the callable nodes of a `DynamicRegimen` | one `rule_kind` for the plan | construction, and `LTMLE.fit` on the raw `regimens=` | `DynamicRegimen.assignment` and `longitudinal_truncation_curve` |
| the `Stochastic` nodes of a `DynamicRegimen` | each node's `density_kind` | construction, and `LTMLE.fit` on the raw `regimens=` | `resolve_plans` and `longitudinal_truncation_curve` |
| `density` of a user-written `Intervention` | a `density_kind` attribute, `None` when absent | the `TMLE` fit | each retarget and `RegimeSet.evaluate` |

A plan of labels alone needs no declaration. A callable written inline in `regimens=` carries none,
so `LTMLE.fit` refuses it. `cleverly._declarations.FunctionDeclaration` shares the check and the
refusal texts of every declaration. `Rule` and `DynamicRegimen` share one rule declaration. The
simulated-confounding replay freezes each regime with the `density_kind` of its source, and never
with a literal `"known"`.

A longitudinal MSM keeps evaluated arrays and a `functions_kind` marker. Its evaluator writes
`"known"` only after the source design and weight declarations pass. A projection built by hand
has no marker, and it refuses truncation replay.
[RM28](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm28-declared-densities-of-user-written-interventions) records the rule.

A learned rule is not a declared rule. `TMLE(learned_rule=LearnedRule())` and `LearnedRuleValue`
estimate its fold-average value on a parameter axis of their own, `"learned_rule"`, which every
axis-keyed consumer decides explicitly. The table gives the rules the fit keeps.

| rule | how the fit keeps it |
| --- | --- |
| the rule of a row comes from a fit that never saw the row | the rule reads the out-of-fold outcome regression of the row's fold |
| one pooled fluctuation | `cv_evaluation=True` with `targeting_scheme="pooled"`; every other scheme refuses before any learner |
| the estimate is the `1/V` average of the fold plug-ins | the fold-evaluated report, with each fold's curve centred at its own estimate |
| no row is refitted | `refute`, the full-refit bootstrap and `repeats` above 1 refuse, because a refit relearns the rules. Every sensitivity analysis refuses too, because no derivation for this target was reviewed |

[Learned rules](technical-reference/point-treatment-tmle.md#learned-rules) states the contract, and
[RM30](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm30-learned-policy-value-evaluation) records its delivery. *Reconsider when* the
package adds another learned-policy target
([X11](roadmap.md#x11-learned-policy-follow-ups)) or a population-law-dependent intervention
function.

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

Only the version that wrote a saved artifact supports it. `result.save()` writes one gzip stream
that holds a header pickle and then the result pickle. The header records `cleverly.__version__`.
On a different version, `cleverly.load()` reads the header and warns with `VersionMismatchWarning`
before it unpickles the result. It then loads the artifact as saved, and it runs no migration,
backfill, or re-stamp.

The load reads the stream to its end, so gzip checks the CRC and the length. The load refuses a
stream with data after the result. `result.save()` writes a temporary file beside the destination and
renames it, so a failed save keeps an earlier artifact at that path.

Development snapshots between two releases share one version string, so the check does not see a
change between them. `cleverly._saved_version` holds the rule, and
`tests/unit/test_serialization.py` checks it. *Reconsider when* the project leaves alpha.

A saved object round-trips the records it holds, and never a value it derived from them. Both
`TMLEResult` and each assessment facade drop every memoized value before joblib writes the artifact.
The shared filter `without_memos` reads the `cached_property` descriptors on the owning class, and
each `__getstate__` applies it. A stored memo records a conclusion and not the question that
produced it, so no reader can check the memo after a load. *Reconsider when* a derived value costs
more to recompute than a load may spend.

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

`run_all` sorts each included capability row into one of three execution classes.

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

A row that the request refuses differs from the row that the fit alone gives. It records the
request's arguments with the signature defaults bound. It carries no next step, because its
sentence names the refused value.

A missing required argument or cost opt-in is `deferred`, because the caller can make the operation
run. A missing method, derivation, replay artifact, or supported requested variant is `unavailable`.
An invoked operation that raises a capability refusal is also unavailable.

The deferral gate reads the capability status, not the supplied argument names. An argument that is
present with the value `None` defers the same operation as an absent argument. The E-value defers on
`estimand=None`, which is its public default. On the four rows of the request rule below, a value
that the call refuses reads `unavailable`.

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

A row whose call refuses some values of one argument resolves each request from the predicate that
the call raises from. The facade's `_gated_map` holds each row after the method and replay gates,
before any request. Its `_request_gated` then resolves one request on top of that map, through the
gate method that the class table `_request_gates` names for the operation. The bare row, a combined
report, and a direct call each resolve their own arguments. The bare row and a combined report
apply the estimand gate first. A direct call does not, because it raises its own refusal of the
default estimand.

| request | row |
| --- | --- |
| the argument is omitted, and the default runs | unchanged |
| the argument is omitted, the default is refused, and another value runs | `deferred` on that argument, with the call's sentence |
| the argument is omitted, and no value runs | `unavailable`, with the default's sentence |
| the argument has a value that the call refuses | `unavailable`, with the call's sentence |

The helper `_argument_resolved` applies this rule to four rows: `truncation_curve` on `mechanism`,
`refute` on `tests`, `benchmark` on `covariates`, and `simulated_confounding` on `estimand`. The
`tipping_gamma` gate refuses `use_ci=True` on a fit that supplies no inference, and it leaves every
other request unchanged. A fit-wide refusal stays in one ordered table, which the row and every
entry point read.

The sweep in `tests/unit/test_capability_row_sweep.py` asks every row of each kind of fit in
`KINDS`. Each row that a request resolves as available must run, and no row may decline or stay
deferred. Each committed mutation in `MUTATIONS` makes the sweep fail on the kinds that it names,
and on no other kind. On each named kind, one problem matches the signature of the mutation. An
unchanged control, M0, passes.

Availability is authoritative before execution. Each capability row names the `Replayability` field
it needs in `requires_replay`, and the shared base applies that gate to every row. A facade may not
patch one row by name. `refute` read its slot only while running, and so reported `available=True`
on a result that carries no estimator.

A replay slot reads the checks that its call runs before any learner. The slot `refit_nuisances`
reads `TMLE._refit_configuration_refusal`. That method runs `_configured_for_refit` and then
`_preflight_fit_configuration`, which includes `_resolve_estimands_for_data` with every subclass
override and the remaining configuration guards. The chain fits nothing, so the method reads
each `ValueError` of the chain as a refusal. `CapabilityError` and `DataError` are `ValueError`,
and so is a malformed setting that a copied estimator can carry. Any other exception is a defect
and propagates.

A result whose estimator holds a configuration that `refit()` refuses keeps
`retarget_cached_nuisances` and loses `refit_nuisances`, with the code
`point_replay_refit_configuration`. Every refit asks
`_configured_for_refit` for the estimator that it fits, and an override adapts it for an added
covariate. The override returns a copy, so a refit never changes the fitted estimator.

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

A modified treatment policy is a map of the natural dose, and `policies=` is its axis. Every
policy class resolves to deterministic branches with known probabilities, through
`policy_branches` for a continuous dose and `discrete_assignments` for a categorical one. The
ratio, the support report and the longitudinal node read those branches and nothing else. A new
class adds a branch type there, not a new axis. `RiskRatioTilt` differs in its report only. It
reads `g`, so it reports on the `rr_tilt` axis. `TMLE` refuses a tilt beside other policies.

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

Resolve dynamic rules and policy densities once, before fitting, into the plan's assignment matrix
and policy arrays. A rule receives only the history available at its node, and a policy density
receives that history and the earlier treatments. Downstream mechanism, support-mask,
outcome-regression and clever-covariate logic must read the same resolved plan. Outcome designs
contain covariate history and the arms of earlier policy nodes, not past treatment columns that are
deterministic under the resolved plan. At a policy node the design also holds the current arm, and
the node carries the policy-weighted mean of its per-arm predictions. A plan without a policy node
runs the arrays of a deterministic plan, and `tests/unit/test_influence_gateaux_longitudinal_policy.py`
pins a one-hot policy to its rule bit for bit.

A modified treatment policy node reuses the per-arm carry of a policy node. At a categorical node
its columns are the levels. At a continuous node its columns are the randomizer branches, and the
node's treatment factor in the cumulative product is the density ratio. `g_bounds` therefore
bounds only the censoring and categorical factors. The fit attaches the ratio to the plan after
the mechanism fit. A frozen plan carries it, so a replay reads the same ratio.

A cross-fitted per-regimen fit targets after its folds, not inside them. Each fold runs an
untargeted backward regression sequence on its training rows. One pooled fluctuation per node then
targets the out-of-fold predictions over every follower, with the out-of-fold cumulative mechanism
in its loss weight. This is Section 5.2, Steps 1 to 4, of Díaz, Williams, Hoffman and Schenck
(2023). The proof of their Theorem 3 needs each fold's fit to be fixed given its training rows.
So no pooled coefficient returns to a fold regression, and the per-regimen fit divides by the
out-of-fold mechanism alone. `n_folds=1` keeps the canonical single-fold recursion, which carries
each targeted prediction back. `tests/unit/test_pooled_longitudinal_targeting.py` holds the
longhand checks and the mutations. *Reconsider when* a published result certifies a fold-local
update for this plug-in estimator, or when the sequential doubly robust estimator ships under
[X4](roadmap.md#x4-sequential-doubly-robust-longitudinal-estimation).

Longitudinal MSMs are projections over regimen/horizon cells. Their fluctuation is pooled and the
backward recursion proceeds in lockstep over nodes; the horizon belongs in the design and each
cause receives its own projection while sharing nuisance fits. Under cross-fitting the fold
recursions run per cell and stay untargeted, and only the pooled update is lockstep. No pooled
coefficient and no working-model coefficient returns to a fold regression. A policy cell moves its
per-arm predictions in the same stacked fluctuation, along its own block of the design, and carries
their policy-weighted mean.

For survival, a unit experiencing the event at node `t` belongs in node `t`'s event regression.
The event-free state is part of history, not an intervened mechanism factor. Reports store risk;
survival is a derived scale, with contrasts transformed by sign rather than by `1 - estimate`.

For competing risks, use a cause-specific event numerator with the all-cause survival factor.
Competing events affect the at-risk population but do not add a denominator factor unless a new
estimand explicitly intervenes on them. Cause-specific estimates need not be renormalized to a
simplex.

A held design has one treatment decision, at node 1, held over every node. Each later node is an
identity node. Its plan value is the observed node-1 value, its factor is exactly one in the ratio
numerator and denominator, and it fits no model. It is never a policy node or a modified treatment
policy node, and it adds no block to a mechanism or outcome design: there is one block per
decision. Regimens resolve on the decisions, so a rule or a policy is evaluated once, on the node-1
history. `tests/unit/test_point_policy_mtp.py` makes the held nodes policy nodes and sees the
Gateaux check fail. `tests/unit/test_point_survival_mutations.py` re-evaluates a rule at every node
and sees the estimate move.

A censoring node at which no eligible unit is censored has the fixed factor one and fits no
learner. The eligible rows are the rows at risk before the node with positive weight. A
cross-fitted training fold with no censored unit predicts one, which is its empirical rate. The
nuisance report shows an omission in place of a model row. `survtmle` sets `G_dC = 1` at `t = 1`
for the same case. `tests/unit/test_no_censoring_node.py` removes the guard and sees the solver
error return.

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
executes the registered documents' fences and the tutorial notebooks' code cells. It runs three
gates, and the table states what each one covers.

| gate | documents | sample size | what it asserts |
| --- | --- | --- | --- |
| `test_every_nonsemantic_example_runs` | entries in `PRELUDES`, none of them tutorials | shrunk to `SMALL_N` | nothing raises |
| `test_tutorial_semantics_at_documented_size` | every tutorial under `docs/examples/`, as Markdown or as a notebook, offline | the size the page prints | nothing raises, then one reviewed callback checks that page |
| `test_every_narrated_decimal_matches_a_stored_output` | every tutorial notebook | none, it reads stored outputs | each decimal in the prose rounds from a stored output |

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

The gate is bounded by review rather than by a heuristic. Each callback lives in its own module
under `tests/unit/tutorial_semantics/` and not in the document, so a rewritten page cannot grant
itself a numeric gate. One module per tutorial lets several tutorials change at once without an
edit to a shared registry. `test_every_tutorial_has_one_semantic_assertion` discovers the
tutorials and pins the modules to them. The documented-size gate is their only runtime pass, which avoids repeating each
fit at the smoke size. *Reconsider when* a callback needs a looser tolerance to keep passing. A
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
