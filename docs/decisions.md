# Standing decisions

Active project decisions that should survive individual implementations and roadmap revisions.
They are not permanent architecture: each remains in force until its stated reconsideration
condition is met. The linked reports and invariants carry the detailed evidence; the roadmap is
reserved for proposed work.

| decision | rationale and evidence | reconsider when |
| --- | --- | --- |
| Keep production code pure Python and `numba` benchmark-only | Nuisance fitting dominates representative workloads, while competent NumPy implementations removed the apparent advantage in the clearest candidate kernels. See the [benchmark verdict](benchmarks/README.md#current-verdict) and focused reports. | A competent compiled implementation wins materially in a full supported workload, including compilation, memory, data movement, packaging, and maintenance cost. |
| Keep internal tabular arithmetic in NumPy | The dataframe boundary is a negligible share of a fit, supported learners consume NumPy arrays, and the public dataframe contract is already isolated through narwhals. See the [dataframe invariants](architecture-invariants.md#dataframes-and-labels). | A supported workload becomes dominated by joins, grouping, IO, or conversion rather than estimation. |
| Parallelize across folds and learner candidates; run individual nuisance fits single-threaded by default | Nested model parallelism oversubscribes small fits, and repeatedly constructing the thread-pool controller was itself a major cost. See the [parallelism invariants](architecture-invariants.md#parallelism-and-performance) and [thread-limit profile](benchmarks/thread_limit_profile.md). | A measured workload benefits from giving one model the machine. Callers can already opt out with `set_thread_limit(None)`. |
| Validate derivations independently; use cross-language comparison only as a bounded secondary check | Implementations descended from the same source can share transcription errors, while derivative, exact-law, remainder, mutation, and score checks fail against distinct error classes. The `LTMLE` fixture is the scoped exception because it pins cumulative-bound placement and a nonzero finite-sample targeting path that exact laws at `epsilon=0` cannot see. See the [evidence index](evidence.md). | Another named blind spot is demonstrated, the compared implementations target the same estimand, and the comparison has predetermined pass/fail actions. |
| Keep generated benchmark results out of Git | Timings describe recorded hardware and environments rather than timeless package facts. The benchmark harness records environment metadata and the reviewed reports retain durable conclusions. See the [benchmark documentation](benchmarks/README.md). | Never for unlabelled raw results. A durable conclusion may enter a reviewed report when it includes reproducible commands and environment metadata. |
| Separate feature selection from statistical certification | Evaluating a configuration on the draws that selected it makes the reported result selection-dependent. | A study performs no data-dependent selection or uses disjoint selection and certification cohorts. |

When a condition is met, update the implementation, its independent evidence, and this register in
the same change. Superseded rationale belongs in Git history or the underlying evidence report,
not in the roadmap.
