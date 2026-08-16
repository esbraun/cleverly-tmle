# Benchmarks

How to measure performance work in `cleverly`, what the current evidence says, and which focused
reports support it. Raw results are machine-specific and belong in the git-ignored
`benchmarks/results/` directory.

## Current verdict

`numba` remains a benchmark-only dependency; nothing under `src/` imports it. Nuisance estimation
dominates representative fits, while the clearest apparent compiler wins disappeared after the
numpy baseline stopped expanding signs unnecessarily, stopped re-deriving cluster encodings, and
stopped rebuilding longitudinal masks quadratically. The largest observed DR-TMLE improvement
instead came from reusing `threadpoolctl`'s controller.

This verdict is deliberately revisable. A production compiled path should be reconsidered when a
competent implementation wins materially in a full supported workload, including compilation,
memory, data movement, packaging, and maintenance cost. HAL is the clearest known workload likely
to meet that condition. The [standing decisions](../decisions.md#standing-decisions) state the full
reopening criteria.

## Durable findings

| area | result | evidence |
| --- | --- | --- |
| nuisance thread limits | caching the controller made entry 59× cheaper and removed 49% of a measured DR-TMLE `retarget` | [thread limiter](thread_limit_profile.md) |
| Rademacher multiplier bootstrap | a bounded numpy buffer was 3.4–3.9× faster and reduced the measured `n = 10⁶` allocation from 1,881 MB to 92 MB without changing the seeded stream | [bootstrap](bootstrap_numpy.md) |
| cluster aggregation | once cluster labels are densified at ingestion, the compiled advantage is negligible or negative at the measured sizes | [cluster aggregation](cluster_integration.md) |
| longitudinal masks | carrying prefix state changes mask construction from `O(T²n)` to `O(Tn)`, but the term is only 0.06% of the measured fit | [longitudinal masks](longitudinal_masks.md) |
| C-TMLE selection cross-fitting | ten implicit inner folds made the guide-size GLM fit about 3.9× slower; an explicit two-fold inner split preserved the fold/full-fit construction at 1.56–1.62× baseline | [C-TMLE nested cross-fitting](ctmle_nested_crossfit.md) |

The benchmark harness under `benchmarks/numba/` retains the wider kernel suite, correctness gates,
memory measurements, compile amortization, core-count controls, and post-nuisance pipeline
denominators needed to challenge these conclusions.

## Measurement rules

- Compare against a competent numpy baseline, not merely the current spelling of a function.
- Report the kernel's share of an end-to-end fit. A large ratio on a negligible region does not
  justify a runtime dependency.
- Measure with the intended learner preset. `library="glm"` is useful as a stress case because it
  makes package-owned arithmetic unusually visible; `library="default"` is the realistic
  denominator.
- Include compile time, warm and cold behavior, allocation, core count, and environment metadata.
- Treat results as properties of their hardware, dependency versions, configuration, and harness.
  Do not compare raw timings across unlike runs.
- Check numerical equivalence before timing. Parallel results must also be stable across supported
  thread counts.

## Running the benchmarks

```bash
pip install -e '.[bench]'
nox -s bench                    # end-to-end fit shares, learners included
nox -s bench-numba              # isolated post-nuisance kernels
nox -s bench-numba-pipelines    # complete post-nuisance pipelines
```

For a short local kernel run, use `benchmarks/configs/sandbox.json`; the full dispatch uses
`benchmarks/configs/full.yaml`. Generated output goes to `benchmarks/results/` and must include its
environment metadata when shared.
