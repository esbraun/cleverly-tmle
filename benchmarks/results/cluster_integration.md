# Cluster aggregation, in numpy — and what is left for a compiler

`findings.md` §2.2 calls `cluster_sums` "the largest ratio in the suite, mostly from serial
numba": 5.5–10.2× serial, 9.5–20.7× on four threads. It names the two costs correctly —
`np.unique` sorts labels that only need hashing, and one `np.bincount` per estimand re-reads
the same index vector — and then reaches for a compiler for both.

The first of those was already paid for, somewhere else, once.

> Measured on the four-core Intel Xeon @ 2.80 GHz container this repository's cloud sessions
> run in, `/proc/loadavg` under 0.6, Python 3.11, numpy 2.4.6, numba 0.66.0. Medians of
> interleaved repetitions, every kernel compiled before any measurement.

## 1. The labels reaching `cluster_sums` are already dense

`cleverly.data.validate.encode_clusters` maps the identifiers onto contiguous codes when the
container is built, and `CausalData.subset` re-derives them. `LongitudinalData` does the same.
So by the time an influence curve is aggregated, `cluster` is `0..C-1` with every code used —
and the `np.unique` inside `cluster_sums` was re-deriving an encoding it had already been
given, on every estimate, every covariance, and every bootstrap replicate.

`cluster_sums` now checks for that in three linear passes — max, min, and a count — and skips
the sort when it holds. The check is exact rather than optimistic, which matters more than it
looks: labels `{0..49, 51}` have a maximum below `n` and a minimum of zero and are *not*
contiguous, and `np.unique` returns one row per observed label where `np.bincount` would
return one per slot. The empty row would go on to change a variance.

Nothing about the output moved. Verified bit-identical against the previous implementation on
contiguous codes, one cluster per row, sparse integers, negative integers, a gap in the codes,
float labels and string labels, for both the 1-d and `(n, m)` shapes.

| n | C | m | before | after | |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 500 | 7 | 0.92 ms | 0.39 ms | **2.38×** |
| 100,000 | 1,000 | 1 | 4.45 ms | 0.32 ms | **13.82×** |
| 100,000 | 1,000 | 5 | 4.44 ms | 1.91 ms | 2.33× |
| 100,000 | 1,000 | 20 | 13.62 ms | 11.01 ms | 1.24× |
| 1,000,000 | 10,000 | 5 | 82.07 ms | 20.40 ms | **4.02×** |

The gain is largest where the sort was the whole cost — few estimands, many rows — and
smallest at `m = 20`, where the per-column passes dominate and the sort never did.

The other numpy arm was measured and not taken. Fusing the `m` passes into a single
`np.bincount` over a flattened `(row, column)` index is 2.3× at `m = 20, n = 10⁵` and **0.53×**
at `n = 10⁶`, where its `8nm`-byte index array stops fitting anywhere useful. One code path, at
the size that matters.

## 2. Through the API, the effect is modest

| | before | after | |
| --- | ---: | ---: | ---: |
| `influence_covariance`, n = 100,000, m = 7, C = 5,000 | 8.0 ms | 3.6 ms | **2.26×** |
| `influence_covariance`, n = 1,000,000, m = 7, C = 50,000 | 113.9 ms | 40.4 ms | **2.82×** |
| clustered `TMLE.retarget`, `glm`, n = 20,000, 5 estimands | 23.0 ms | 23.2 ms | 0.99× |
| clustered `TMLE.retarget`, `glm`, n = 100,000, 5 estimands | 101.8 ms | 91.5 ms | 1.11× |
| clustered nonparametric bootstrap, 15 replicates | 21.79 s | 21.17 s | 1.03× |

At the size a point-treatment fit is usually run, this is invisible — one millisecond of a
twenty-three millisecond `retarget`. It becomes visible at `n ≥ 10⁵` and it is free, so it
stays; but it is not the finding, and quoting the 13.8× from §1 as though it were a fit-level
number would be exactly the error `findings.md` §5 warns about.

The bootstrap row is `findings.md` §8's point restated as a measurement: `run_bootstrap` refits
the whole estimator per replicate, so it is nuisance-fitting-bound and no aggregation kernel
reaches it.

## 3. The benchmark was measuring a copy of the old code

`benchmarks/numba/kernels/clustered.py::numpy_cluster_sums` was a verbatim copy of the shipped
function's body. That is how a benchmark tells its most durable lie: the reference does not
move when the package does, so the compiled kernel goes on being compared against whatever
existed when the benchmark was written. It now **calls** `cleverly.inference.cluster.cluster_sums`.

The fixture had the same problem one level down, and this one is worth spelling out because
the reasoning in its docstring was explicit and wrong:

> The codes are deliberately **shuffled and sparse** … because the production implementation
> calls `np.unique` to densify them and a fixture of `0..C-1` in order would hide that sort's
> cost.

A raw `id` column never reaches `cluster_sums`. `make_cluster` now takes `labels=`, defaulting
to `"encoded"` — the codes the container actually holds — and keeps `"raw"` for the other
question, which is what a caller invoking the public function directly pays.

## 4. Against the codes production passes, the compiled kernel is not the win it was

Same kernels, same box, the only change being which labels the fixture hands them:

| labels | shape | n | C | m | numpy | numba | serial | numba ×4 | parallel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **encoded** | balanced | 100,000 | 1,000 | 5 | 2.07 ms | 2.03 ms | **1.02×** | 2.06 ms | 1.01× |
| **encoded** | balanced | 100,000 | 1,000 | 7 | 3.15 ms | 0.73 ms | 4.31× | 0.50 ms | 6.35× |
| **encoded** | balanced | 100,000 | 1,000 | 20 | 11.05 ms | 1.05 ms | 10.51× | 0.61 ms | 18.15× |
| **encoded** | skewed | 100,000 | 1,000 | 5 | 2.03 ms | 0.76 ms | 2.67× | 0.56 ms | 3.62× |
| **encoded** | balanced | 1,000,000 | 10,000 | 5 | 26.14 ms | 35.47 ms | **0.74×** | 34.14 ms | 0.77× |
| **encoded** | balanced | 1,000,000 | 10,000 | 20 | 121.35 ms | 34.43 ms | 3.52× | 20.37 ms | 5.96× |
| raw | balanced | 100,000 | 1,000 | 5 | 4.57 ms | 0.62 ms | 7.32× | 0.42 ms | 10.99× |
| raw | balanced | 100,000 | 1,000 | 20 | 13.39 ms | 1.05 ms | 12.69× | 0.58 ms | 23.02× |
| raw | balanced | 1,000,000 | 10,000 | 5 | 84.40 ms | 20.87 ms | 4.04× | 18.38 ms | 4.59× |

The `raw` rows reproduce `findings.md`'s 5.5–10.2×. The `encoded` rows are the same question
asked of the input the package actually produces, and there **the compiled kernel is a wash at
five estimands and loses outright at a million rows**. What survives is the estimand axis —
10.5× at `m = 20` — which is the half of the original diagnosis the numpy fix could not take:
fusing `m` passes over one index vector into one pass is a thing a compiler does and numpy does
not.

The package's own default estimand sets are three to seven, and `m = 5` is where this is a
wash. So the honest classification is **retain numpy, unless a caller is aggregating twenty or
more curves at once** — not "strong production candidate", and not `findings.md`'s "adopt
numba (parallel above ~10⁵ rows)", whose crossover was measured against a sort that no longer
runs.

**One row is reproducible and unexplained.** The compiled kernel takes 2.03 ms at `m = 5` and
0.73 ms at `m = 7` on identical labels and 40% more work — confirmed across seeds, orderings
and fresh processes. Whatever the cause (a vectorisation cliff in the inner column loop is the
obvious guess), a kernel whose timing has a cliff between five and seven columns is itself an
argument for measuring the configuration you will run rather than the one that benchmarks well.
