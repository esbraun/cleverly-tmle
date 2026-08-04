# Findings: numba and parallelism after the nuisances are fitted

The recommendation half of the investigation whose profile is
[`candidate_inventory.md`](candidate_inventory.md). Read that first — it is what the work
was sized against, and three of the things it is natural to expect turn out to be false.

> ### How these conclusions changed, and the lesson under all of it
>
> **The corrections below are applied in the body of this document**, section by section,
> so a reader does not need another one open to trust a number here. This table is the
> summary and the history, not a redirect: acting on this document produced
> [`production_plan.md`](production_plan.md) and four changes, and each contradicted
> something it said.
>
> The measurements were reproducible. Several of the **conclusions** were not what they
> should have been, and the reason is one mistake made three times: **a ratio measured
> against the shipped shape is not a ratio against numpy.** Two of the three largest "adopt
> numba" recommendations here dissolved when the numpy side was written properly, and the
> third was a share of the wrong denominator. Before quoting anything below, check what the
> *production* function costs and what a competent numpy version of it would.
>
> | as issued | as it stands | where |
> | --- | --- | --- |
> | `multiplier_bootstrap`: **adopt numba parallel**, 2.4–2.5× serial | The numpy path is now **3.4–3.9×** faster than it was and holds its buffer to a 32 MB budget with a four-replicate floor. The cost was never the draw (2%) but the float64 expansion (89%). A compiled kernel must beat *that*; the one measured for the plan does not. **Unresolved.** | [`bootstrap_numpy.md`](bootstrap_numpy.md) |
> | `cluster_sums`: **adopt numba**, 5.5–10.2× serial | Measured against the codes production actually passes — the container densifies them once — the compiled kernel is **1.02× at five estimands and 0.74× at a million rows**. The 5.5–10.2× was mostly an `np.unique` the package no longer runs. **Retain numpy**, on a fit-level share; the crossover in `m` is not located. | [`cluster_integration.md`](cluster_integration.md) |
> | `ltmle_backward_recursion`: fix the masks, then **adopt numba parallel** | The mask fix is made and is `O(T²n)` → `O(Tn)`, 8.7× on the mask term at `T = 20`. In a **fit** it is 0.06% of the runtime at every `T` up to 40. The recursion's whole package-owned arithmetic is 1.5% of a `glm` fit; `inference` is 20%. | [`longitudinal_masks.md`](longitudinal_masks.md) |
> | §3.2's memory column and §6's thread limiter | `tracemalloc` **does** see numba's allocations, so the caveat under that table is wrong. And the thread limiter is fixed: 59× per entry, **49% of a DR-TMLE `retarget`**. | [`thread_limit_profile.md`](thread_limit_profile.md) |
> | the timing method, described throughout as *interleaved* | It was randomised **block** order: each arm's repetitions back to back, the order of the arms shuffled. The harness now rotates properly, so a rerun is a different instrument — see the provenance note below. | — |

Everything here is **post-nuisance**: learner fits are outside every timed region. That is
what makes `n = 1,000,000` a couple of seconds and the scaling questions answerable by
direct measurement — and it is also why every speed-up below has to be multiplied by §5's
share before it means anything about a fit.

> **Provenance.** Four-core Intel Xeon @ 2.80 GHz container, OpenBLAS 0.3.31 (pthreads),
> Python 3.11, numpy 2.4.6, numba 0.66.0. Regenerate with
> `python -m benchmarks.numba.cli --config benchmarks/configs/sandbox.json`. A four-core
> shared box is a poor place to measure efficiency past two threads and a fine place to
> measure a serial speed-up; `benchmarks/configs/full.yaml` is the sweep to run on a
> machine with cores to spare. Results from different hardware are different measurements.
>
> **And from a different harness.** Every number below was taken in randomised *block*
> order — each arm's repetitions run back to back, the order of the arms shuffled — which
> this document and the harness's own docstring both called interleaving and which is not.
> `measure_interleaved` now rotates properly, so **a rerun is a different instrument rather
> than a replication**, and the ratios here that sit within a few percent of 1.0 were never
> resolved by them. The large ones are not in doubt: block order confounds drift with arm,
> and no plausible drift on this box turns a 10× into a 1×.

---

## 1. The answer in one table

Both speed-up columns are **totals against the shipped numpy path at one core**, so the
second is compilation *and* parallelism together and is not a scaling figure. Same-backend
scaling — `T_candidate,1 / T_candidate,p` and its efficiency — is what
`latest/summary.md`'s per-kernel tables report, and the two must not be quoted for each
other. The `decision` column is the classification `production_plan.md` §1 argues for;
where it differs from the one this table shipped with, the banner above says why.

| kernel | serial vs numpy-1 | 4 cores vs numpy-1 | memory | classification |
| --- | ---: | ---: | --- | --- |
| `multiplier_bootstrap` | 2.4–2.5× | 7.4–7.6× | 427.6 MB → 1.6 MB per call | **unresolved** — the numpy path is now 3.4–3.9× faster than the one measured here |
| `cluster_sums` | 5.5–10.2× | 9.5–20.7× | 1.22× (a hash table where numpy sorts) | **retain numpy** — measured against densified codes it is 1.02x at five estimands, and the crossover in `m` is unlocated |
| `ltmle_backward_recursion` | **2.1–4.7×** | **7.9–15.2×** | 0.61× | **promising prototype** — the region it speeds up is 1.5% of a fit |
| `survival_incidence` | **3.4×** | **9.5×** | 0.87× | **promising prototype**; integrated gain unresolved |
| `one_step_walk` | **2.6–3.0×** | **3.8–4.1×** | 0.50× | **semantic change first** — which estimator is intended when the shrink bound binds |
| `ctmle_candidate_scores` | 2.5× | 9.2× | 0.37× | **defer** — 11 ms of a 199 s fit |
| `msm_gram` *(control)* | **2.9–3.2×** | n/a | **~0** (no intermediates) | **a layout result, not a numba one; see §3** |
| `fused_influence_curves` | 1.3–3.0× | 2.6–4.8× | **2.00×** (a dense `(7, n)` output) | **retain numpy** — the share is a tenth of a tenth |
| `drtmle_reduction_rounds` | 0.97–1.00× | 1.8–1.9× | 0.31× | **retain numpy** — §6 is where its time is, and §6 is now fixed |
| `newton_targeting` *(control)* | 1.1–1.4× | n/a | ~0 | **retain numpy**, as expected |
| `cvtmle_fold_targeting` | 1.4–3.9× | **2.8–13.4×** | 0.25–0.50× | **promising prototype** — task parallelism over the production solver first |

**These verdicts are not always `latest/summary.md`'s, and the difference is deliberate.**
That file is generated: it applies the plan's continuation bars (1.25× serial, 1.5× parallel,
25% memory) to each kernel's own ratios and nothing else. This table multiplies by §5's
share first. So `drtmle_reduction_rounds` clears the parallel bar mechanically and is
retained here anyway, because 1.9× of an arithmetic step that is a minority of a `retarget`
whose majority is a context manager is not worth a dependency. Where the two disagree, the
generated file is the measurement and this one is the judgement.

And the one that is not a compilation question at all:

| finding | magnitude | decision |
| --- | --- | --- |
| `cleverly.learners.thread_limit` builds a fresh `ThreadpoolController` per learner fit | **1.44 ms per entry**; 57% of a DR-TMLE `retarget`, 40% of an LTMLE fit | ✅ **fixed** — 59× per entry, 49% of a `retarget` removed (§6) |

---

## 2. Kernel by kernel

### 2.1 The Rademacher multiplier bootstrap — the clearest adoption, and it was numpy's

> **The cost is not the draw.** Split at `n = 100,000`: `rng.integers` 3.5 ms (2%),
> `np.unpackbits` 1.7 ms (1%), expanding those bits into a 205 MB float64 array 159 ms
> (**89%**), the `dgemm` 12.6 ms (7%). Expanding in place into a reused buffer, with the
> block sized by bytes, is **3.4–3.9×** in numpy alone and holds its buffer to a 32 MB budget —
> faster than the compiled kernel below is here, with no dependency and the seeded stream
> untouched. See [`bootstrap_numpy.md`](bootstrap_numpy.md).

`docs/roadmap.md` already records that this is 92–95% multiplier *generation*, and the
package already took the cheap half of that (packing bits instead of drawing float64
uniforms). What it did not change is the shape: a `(chunk, n)` float64 array is
materialised in order to be consumed once, and then multiplied into `centred` by a BLAS
`dgemm` whose left operand is a matrix of plus and minus ones.

A fused kernel never forms it. Each row draws its bit and is added to or subtracted from an
`m`-vector accumulator, so both the array and the multiply disappear together.

| n | numpy | numba | numba, 4 threads |
| ---: | ---: | ---: | ---: |
| 10,000 | 53.1 ms | 19.8 ms (2.7×) | 7.1 ms (7.4×) |
| 100,000 | 448.6 ms | 174.5 ms (2.6×) | 51.1 ms (8.8×) |

(These are the second of two independent sweeps; the first read 2.5× and 7.6× at
`n = 100,000`. The two agree to within the run-to-run spread of a shared four-core box,
which is itself worth knowing — a ratio quoted to two figures here is not a ratio to two
figures.)

**Blocking is the whole difference and is worth naming**, because the obvious version of
this kernel *loses*. One replicate at a time reads the whole `(n, m)` array once per
replicate — 2 GB of traffic for 250 MFLOP at `B = 500`, `n = 10⁵` — and measured at 405 ms
against numpy's 393. Accumulating 64 replicates per pass reuses each loaded row 64 times
and takes it to 188 ms. That is a blocked `dgemm`, done where the sign draw can be fused
into it.

**The memory argument is separate and may matter more than the speed.** Measured per call
with `tracemalloc` at `n = 100,000`, `B = 500`:

| implementation | allocated per call |
| --- | ---: |
| numpy (the shipped path) | **427.6 MB** |
| numba, serial or parallel | **1.6 MB** |

A **264× reduction**, and it does not grow with `n`: the fused kernel's working set is
`block × m` doubles — 32 KB — and the 1.6 MB is the statistics array and the resampled
standard error the correctness gate needs, neither of which the kernel is responsible for.
The roadmap already names the numpy array as one of two allocations that break before any
arithmetic does, at ≈9.5 GB at `n = 5,000,000`. This is the one result here that changes
what the package *can* do rather than how fast it does it.

(Process peak RSS is not the instrument for this and the harness does not use it: it is a
high-water mark that never falls, so once the first implementation has touched the pages
every later one reads a delta of zero. The numbers above are per-call allocation, taken in
an untimed pass with `tracemalloc` — which sees numba's NRT allocations as well as numpy's;
see §3.2.)

**And the numpy figure has since fallen to a 32 MB buffer budget without a
compiler**, which is what makes this row *unresolved* rather than an adoption: see
[`bootstrap_numpy.md`](bootstrap_numpy.md). The capability claim in the paragraph above
survives; what does not is the attribution of it to compilation.

**Reproducibility is by construction, not by luck.** The draw is a splitmix64 hash of
`(seed, replicate index)`, so a replicate's multipliers do not depend on which thread ran
it or how many threads there were. One thread and four threads give bit-identical answers,
which `tests/unit/test_numba_benchmark.py` pins — and the same test checks the fused
statistic against signs reconstructed in numpy from the same counter, because the
cross-generator quantile gate is a Monte Carlo one and would pass a kernel that dropped a
column and got lucky.

### 2.2 `cluster_sums` — the largest ratio in the suite, and most of it was a sort

Two costs, and only one of them is about compilation. `np.unique` **sorts** to densify
labels that only need hashing, and then one `np.bincount` per estimand re-reads the same
index vector `m` times.

> **The diagnosis was right and the conclusion was not.** The sort was already paid for
> somewhere else: `encode_clusters` densifies the identifiers when the *container* is
> built, so a raw `id` column never reaches this function and the `np.unique` here was
> re-deriving an encoding it had been given. The fixture below feeds it sparse labels on
> the explicit reasoning that "a fixture of `0..C-1` would hide that sort's cost" — which
> is the internal contract inverted. Against densified codes the compiled kernel is
> **1.02× at five estimands and 0.74× at a million rows**; what survives is the estimand
> axis, 10.5× at `m = 20`. See [`cluster_integration.md`](cluster_integration.md); the
> numbers in this section are correct for the labels they were measured on.

| n | m | shape | numpy | numba | numba, 4 threads |
| ---: | ---: | --- | ---: | ---: | ---: |
| 100,000 | 5 | balanced | 5.4 ms | 0.9 ms (6.2×) | 0.6 ms (9.3×) |
| 100,000 | 5 | skewed | 7.0 ms | 0.9 ms (8.0×) | 0.5 ms (12.9×) |
| 100,000 | 20 | balanced | 17.7 ms | 1.9 ms (9.1×) | 0.9 ms (20.6×) |
| 100,000 | 20 | skewed | 18.7 ms | 1.8 ms (10.6×) | 1.0 ms (19.3×) |

**Most of the gain is serial**, which is the useful part of this result: a hash-based
densify and one pass accumulating all `m` columns is 5–10× before any thread is involved.
At `n = 10,000` the parallel arm is a wash or worse — the pool costs more than the work —
so the recommendation is *numba serial always, parallel above ~10⁵ rows*.

**The "improve the numpy instead" arm failed here, and that is informative.** Sorting once
and reducing all `m` columns with `np.add.reduceat` is **slower** than the shipped path
(0.39–0.88× of it): the sort is the expensive half, so doing it once and saving the
repeated `bincount` passes does not pay for keeping it. What is wanted is not a better
sort — it is not sorting.

Two parallel strategies were run because the winner is not guessable. Thread-local
accumulators (`threads × C × m` doubles) win on the balanced and skewed designs alike here;
partitioning over clusters after a counting sort wins at small `n`, and loses badly on a
skewed design (a handful of clusters hold most of the rows, so one thread runs long after
the others are idle) — which is why `shape` is a swept dimension and not a fixture detail.

### 2.3 The LTMLE backward recursion — where the algorithm matters more than the compiler

> **The mask fix is made, and it is 0.06% of a fit.** The ratios below are of the
> *cached-nuisance recursion*, which excludes the learner fits by construction. Measured
> through the API with explicit phase timing, mask construction is 0.06–0.13% of a `glm`
> longitudinal fit at every `T` from 2 to 40, and the recursion's whole package-owned
> arithmetic is 1.5%. The fix is still right — `O(T²n)` → `O(Tn)`, 8.7× on the mask term at
> `T = 20` — and it is not a fit-level speed-up.
> See [`longitudinal_masks.md`](longitudinal_masks.md).

| n | T | numpy | numpy, prefix masks | numba | numba, 4 threads |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 5 | 33.9 ms | 31.7 ms (1.07×) | — | 3.9 ms (8.6×) |
| 100,000 | 5 | 336 ms | 336 ms (1.00×) | — | 41 ms (8.2×) |
| 10,000 | 20 | 209 ms | 122 ms (**1.71×**) | 53 ms (3.9×) | 17 ms (12.7×) |
| 100,000 | 20 | 4,083 ms | 1,673 ms (**2.44×**) | 856 ms (4.8×) | 253 ms (16.1×) |

`at_risk(t)` is `uncensored_through(t-1) & followed_through(t-1) & event_free(t-1)`, and
`followed_through` is itself a loop of `t` boolean `&` passes. Called at every node, that is
`O(T² n)` — invisible at the `T = 2` the package's own fixtures use, and **2.4× of the
recursion at `T = 20`**. Carrying the running masks down the nodes instead of rebuilding
them is `O(T n)`, computes the same masks (they are a prefix scan of the same conjunction),
and needs no compiler.

So the order matters: **fix the masks first, then compile.** Adopting the compiled kernel
without the mask fix buys a compiled version of redundant work, and the reported 16× would
be against a baseline nobody should be running.

Regimens are the parallel axis and they are genuinely independent once the node predictions
exist — separate masks, separate cumulative product, separate fluctuation. The recursion
over `t` is left alone: node `t` regresses on what node `t+1` produced, and a scan
formulation for a *targeted* recursion does not exist. Parallelising it would be a
different estimator, not a faster one.

### 2.4 Survival and competing risks — the same recursion, run `T(T+1)/2` times

At `n = 20,000`, `T = 10`, one regimen pair, every horizon: numpy 778 ms → numba 213 ms
(3.7×) → numba on four threads 77 ms (**10.2×**).

Sharing the horizon-independent masks and cumulative product across the horizons — they do
not depend on the horizon at all — is 1.19× on its own. The parallel arm rebuilds them per
cell deliberately: a flat `prange` over `(regimen, cause, horizon)` needs no barrier, and at
`T` nodes the mask pass is `O(T n)` against the cell's `O(T² n)`, so paying it back is what
the flat axis buys. The serial kernel keeps the shared form, so the report can separate the
two costs.

This is the flavour where the compiled kernel matters most in absolute terms: a real
survival fit runs `T(T+1)/2` node regressions per regimen per cause, and the package-owned
half of each of them is what this measures.

### 2.5 The one-step walk — the plan's high-priority target, and the answer is a rewrite first

| n | numpy | numpy, deferred arms | numba, deferred | numba parallel, deferred |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 54.5 ms | 35.6 ms (**1.53×**) | 19.1 ms (2.86×) | 9.6 ms (5.68×) |
| 100,000 | 386 ms | 222 ms (**1.74×**) | 136 ms (2.83×) | 95 ms (4.08×) |

**Compiling the walk as written buys nothing: 1.00× at `n = 100,000`.** Measured, not
inferred — the fused kernel and the numpy path are within noise of each other, because both
are bound by the same `3n` transcendental evaluations per step and numba's scalar `exp` is
no worse than numpy's vectorised one (it is in fact 2.7× faster per element; the loop
overhead around it eats the difference).

What buys something is noticing that **the walk's score reads only the fit at the observed
treatment**. The `K` counterfactual arms are updated on every trial step and are not read
until the walk ends; since `logit` is additive along the submodel, updating them once at the
accumulated `epsilon` is the same array — measured at `2.6e-15` after fifty steps — for
`1/(K+1)` of the transcendental work. That is 1.5–1.7× from numpy alone, and it is what
takes the compiled kernel from 1.0× to 2.8×.

**The equivalence is conditional and the condition is stated.** `shrunk(alpha)` is applied
after each step, so once an arm's prediction is pinned at the bound the incremental path
clamps repeatedly and the deferred one clamps once. Under good overlap it does not bind and
the two agree to rounding, which `tests/unit/test_numba_benchmark.py` pins. Under a severe
positivity regime they can diverge, and any adoption of this has to decide which of the two
is the intended estimator rather than treating it as an optimisation.

Rows are the only parallel axis: each step's direction is the score at the fit the previous
step produced. Efficiency is poor (4.1× on four cores at `n = 10⁵`) because each step ends
in a cross-thread reduction over a `p`-vector, and `p` is two.

### 2.6 CV-TMLE's fold loop — task parallelism, and the arm that regressed

`TMLE._solve_by_fold` is a serial Python `for` over folds that are independent by
construction: a fold's `epsilon` is fitted only against rows whose nuisance predictions came
from models that never saw them. Folds cut each task's rows by `1/F` while giving the task
axis more tasks, so within-fold and across-fold parallelism cross somewhere, and where is a
property of the machine rather than of the algorithm. Both were run.

| n | folds | numpy | `numpy_threads` @4 | numba | `numba_parallel` @4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 2 | — | — | 1.83× | 3.57× |
| 100,000 | 2 | — | — | 1.65× | 3.21× |
| 10,000 | 10 | — | — | 3.03× | 9.54× |
| 100,000 | 10 | 64.2 ms | 54.6 ms (1.18×) | 1.69× | 9.7 ms (**6.30×**) |
| 10,000 | 20 | — | — | 4.12× | 14.25× |
| 100,000 | 20 | — | — | 1.47× | 4.98× |

**The thread-pool arm is the finding.** Running the *unchanged numpy* fold bodies over a
`ThreadPoolExecutor` reaches 1.46× on two cores and then **regresses to 1.18× on four** —
the GIL, held by the Python between the BLAS calls, which at these fold sizes is most of the
per-fold time. Task parallelism over numpy bodies is not the cheap win it looks like. The
compiled `prange` over folds, whose bodies hold no GIL at all, reaches 3.82× at **0.95
efficiency** on the same configuration.

The other thing the sweep shows is that the fold count is not monotone: the compiled
parallel gain peaks around ten to twenty folds at small `n` and falls at `n = 100,000, F =
20`, where each fold is large enough that the serial stitch — `O(n)` fancy-indexing that
every arm still has to do — starts to bound the whole thing.

Note what this axis is *not* about. The per-fold body is the Newton solve, which §3 confirms
compilation does not help on its own; the gain here is scheduling, not arithmetic.

### 2.7 CTMLE and DR-TMLE — measured, and deferred

`ctmle_candidate_scores` is 2.6× serial and 9.7× on four threads, and it does not matter:
the post-selection `retarget` is **11 ms of a 190-second fit**, and the candidate search is
candidate-*fitting*-bound. The kernel stays in the suite because it is what would matter if
candidates were ever cheap — a linear-model library, or a sweep over an already-selected
path — and because parallelising it is only valid for a *pre-enumerated* candidate set. A
forward-selection path chooses candidate `k+1` after seeing candidate `k`; parallelising
that would compute a different path, and the benchmark does not.

`drtmle_reduction_rounds` is the one kernel where **numba is slower than numpy serially**
(0.93× at `n = 10⁴`, 1.04× at `n = 10⁵`). The rounds are strictly sequential — each reads
the mechanism the last one tilted — so the only axis is the arms, which is two wide on a
two-arm fit; 1.6–1.9× on two cores, nothing above. And the arithmetic is not where a
DR-TMLE `retarget`'s time goes anyway: §6 is.

### 2.8 The fused multi-estimand influence curves — the plan's headline hypothesis, measured

1.1–2.8× serial and 1.6–4.1× on four threads, rising with the estimand count exactly as the
fusion argument predicts. And it does not matter, for the reason the inventory gave before
any of it was written: `counterfactual_means` and `_conditional_effects` together are **2–4
ms of a 28.6 ms `retarget`**. The estimand count is a proxy for the *target-group* count,
and each group's cost is its own fluctuation solve, not its curve.

A perfect fusion returns a tenth of a step that is itself 5.5% of a `glm` fit and 0.15% of a
`default` one. **Retain numpy.**

Kept in the suite, because "we fused it and it did not matter" is a usable finding and "we
assumed it would not matter" is not — and because it is the natural place to check that the
identity `IC_ate == IC_ey1 - IC_ey0` survives fusion, which it does exactly.

---

## 3. The controls: one that held, one that did not, and one that makes the rest mean anything

### 3.0 Threaded BLAS — the arm that attributes the speed-ups

A parallel speed-up is only a statement about the *kernel* if the reference is known not to
get one from the same cores. So the numpy reference is re-run unchanged with BLAS given
every core (`numpy_threaded_blas`), and at `n = 100,000`:

| kernel | numpy, BLAS=1 | BLAS=2 | BLAS=4 |
| --- | ---: | ---: | ---: |
| `fused_influence_curves` (7 estimands) | 7.11 ms | 7.51 ms | 6.95 ms |
| `cluster_sums` (m=5, balanced) | 5.36 ms | 5.55 ms | 5.39 ms |

**Flat, to within the noise.** These kernels are ufunc and indexing work, not `dgemm` work,
so BLAS has nothing to thread. Every parallel number elsewhere in this document is therefore
attributable to the compiled kernel rather than to the cores it was given — which is the
whole reason this arm exists, and why a run that omitted it would be quoting a ratio it had
not earned.

### 3.1 The negative controls, one of which was not

#### `newton_targeting`

`newton_targeting` behaves exactly as `benchmarks/bench_tmle.py` said it would: **1.4× at
`n = 10⁴`, 1.18× at `n = 10⁵`**, falling with size and below the 1.25× continuation bar at
the larger one. Its inner work is `x @ eps`, `x.T @ (w r)` and `x.T @ (x v)` — BLAS calls —
plus one vectorised `expit`. A compiler pays where the interpreter is in the inner loop, and
here it is not. **Retain numpy**, confirmed on this harness rather than quoted from another.

#### `msm_gram`

`msm_gram` was included on the same reasoning and **the reasoning was wrong**. The
four-operand `einsum('ijp,ijq,ij,i->pq', …, optimize=True)` is **3.1–3.4× slower than a
hand-written loop**, at both sizes. `optimize=True` fixed the catastrophic case — the
roadmap records the unoptimised spelling at 14× slower — but what it produces is a pairwise
contraction with materialised intermediates, not the single fused pass the arithmetic
allows. Since the intermediates are `(n, K, p)`, that is also an allocation the loop does
not make.

This is why the controls are in the suite. A benchmark that only runs the kernels it expects
to win cannot distinguish "numba helps here" from "the harness flatters numba", and it
cannot catch a case like this one, where the *reason* a kernel was dismissed turns out not
to hold. The projection is still ~1% of a fit, so this is not an adoption recommendation —
it is a correction to the inventory's reasoning, and a note that the identity-link closed
form's bit-for-bit regression pin (`test_the_identity_link_is_the_closed_form_bit_for_bit`)
would have to be reckoned with before anything moved.

---

## 3.2 Memory, across the whole suite

Per-call Python-level allocation, best compiled implementation against the shipped numpy
path. The bootstrap is the outlier and the rest are not uniform, which is the point of
reporting the column rather than one row of it.

| kernel | n | numpy | best numba | ratio |
| --- | ---: | ---: | ---: | ---: |
| `multiplier_bootstrap` | 100,000 | 427.55 MB | 1.62 MB | **0.004×** |
| `msm_gram` | 100,000 | 25.61 MB | ~0 | **0.00×** |
| `newton_targeting` | 100,000 | 4.00 MB | ~0 | **0.00×** |
| `cvtmle_fold_targeting` | 100,000 | 9.66 MB | 2.40 MB | 0.25× |
| `drtmle_reduction_rounds` | 100,000 | 10.41 MB | 3.20 MB | 0.31× |
| `ctmle_candidate_scores` | 50,000 | 3.26 MB | 1.20 MB | 0.37× |
| `one_step_walk` | 100,000 | 9.61 MB | 4.80 MB | 0.50× |
| `ltmle_backward_recursion` | 100,000 | 33.42 MB | 20.50 MB | 0.61× |
| `survival_incidence` | 20,000 | 11.33 MB | 9.80 MB | 0.87× |
| `cluster_sums` | 100,000 | 4.11 MB | 4.99 MB | **1.22×** |
| `fused_influence_curves` | 100,000 | 3.20 MB | 6.40 MB | **2.00×** |

Three groups, and the third is the one a reader should not miss.

**Allocation falls where the numpy path materialises intermediates it consumes once** — the
bootstrap's multiplier matrix, the Gram contraction's einsum intermediates, the walk's
per-step arrays. `newton_targeting` and `msm_gram` reaching ~0 is the fused loop
accumulating into a `(p, p)` scratch and nothing else; that is real, and it is also the two
kernels whose *time* is a wash or does not matter.

**Allocation is unchanged where the output dominates** — the recursions carry a targeted
prediction per node per regimen either way.

**The instrument sees both sides.** `tracemalloc` traces all three CPython allocator
domains and numba's NRT allocates through one of them, so a compiled kernel's scratch is
counted here exactly as numpy's is — including a `prange` kernel's per-thread block, which
`numba_cluster_sums_threadlocal` shows directly: 40.0 MB at `C = 50,000, m = 20` on four
threads against the serial kernel's 9.99 MB, a difference of 30 MB against a 32 MB
thread-local accumulator. An earlier revision of this section said the opposite, on the
strength of a docstring in `timing.py` that was wrong; both are corrected.

**Allocation goes up in two kernels, and both are honest costs rather than defects.**
`cluster_sums`'s hash table is `2n` int64 slots where `np.unique` sorts in place, so the
compiled kernel trades 1.2× the memory for 5–10× the speed. `fused_influence_curves`
allocates a dense `(7, n)` output where the numpy path builds only the estimands asked for
— at seven estimands that is 2×, and it is the fusion's own design: computing every curve
in one traversal means allocating every curve. Neither is a reason to reject the kernel;
both are reasons the column is reported per kernel and not summarised into a claim that
compiled code allocates less.

---

## 4. Compilation is not free, and the break-even is the number to quote

Measured in a fresh process per kernel (`--cold-compile`), because numba caches a compiled
signature for the life of the process and "the first call" is only the first call once.

| kernel | implementation | compile | break-even calls |
| --- | --- | ---: | ---: |
| `survival_incidence` | `numba_parallel` | 4.62 s | **18** |
| `ctmle_candidate_scores` | `numba_parallel` | 1.54 s | **56** |
| `multiplier_bootstrap` | `numba_parallel` | 2.02 s | **136** |
| `one_step_walk` | `numba_deferred_arms` | 3.09 s | **251** |
| `ltmle_backward_recursion` | `numba` | 2.56 s | **306** |
| `drtmle_reduction_rounds` | `numba` | 1.35 s | 807 |
| `cvtmle_fold_targeting` | `numba_parallel` | 9.33 s | 2,087 |
| `msm_gram` | `numba` | 0.89 s | 2,780 |
| `fused_influence_curves` | `numba` | 0.85 s | 5,684 |
| `cluster_sums` | `numba` | 1.34 s | 12,732 |
| `newton_targeting` | `numba` | 6.17 s | 34,691 |

**Break-even is at the *small* fixture (n = 2,000) and is therefore an upper bound**: the
saving grows with `n` and the compilation does not. Read it as a ranking rather than as a
count — `survival_incidence` pays for itself in eighteen calls at two thousand rows and in
far fewer at a hundred thousand, while `newton_targeting`'s thirty-five thousand is another
way of saying the control is a control.

The practical consequence: **compilation is amortised by the repeated workloads and by them
alone.** A single fit calls each of these once or a few times and would pay 1–9 seconds for
it. A sensitivity sweep, a simulation study, a bootstrap or a survival curve calls them
hundreds of times, and there the compile is invisible. That is the same conclusion §5
reaches from the other end.

Two compile times are worth flagging on their own. `cvtmle_fold_targeting` at 9.3 s and
`newton_targeting` at 6.2 s are the two kernels that call `np.linalg.solve` inside a jitted
function, which drags numba's LAPACK bindings into the compilation.

---

## 5. The denominator: what any of this is worth

Measured through the shipped API at `n = 20,000`, `library="glm"`, with the fit outside the
timed region (`benchmarks/results/latest/pipelines.md`):

| scenario | fit | post-nuisance | share of the fit |
| --- | ---: | ---: | ---: |
| `tmle_iterative` (Newton, 7 estimands) | 0.489 s | 25.8 ms | **5.3%** |
| `tmle_one_step` (the universal walk) | 0.614 s | 181.7 ms | **29.6%** |
| `cvtmle_10folds` | 1.694 s | 111.3 ms | **6.6%** |
| `ltmle` (2 nodes, 2 regimens)¹ | 1.992 s | 1.147 s | **57.6%** |
| `survival` (per-horizon passes)¹ | 2.366 s | 1.331 s | **56.2%** |
| `sensitivity_grid25` (one fit, 25 retargets) | 0.427 s | 0.639 s | **149.7%** |
| `drtmle` (one `retarget`) | 10.519 s | 16.193 s | **153.9%** |
| `ctmle` (post-selection `retarget`) | 199.457 s | 11.0 ms | **0.01%** |

¹ `LTMLE` refuses a `retarget` by design — `g_bounds` enters every earlier node's
pseudo-outcome through the recursion — so its post-nuisance half cannot be *called*, only
separated inside a profile. The figure is the fit net of scikit-learn, LightGBM, joblib,
scipy and threadpoolctl, which is why it is large: the biggest single line inside it is the
Rademacher multiplier draw at 14% of the whole fit, and §6's thread limiter is *excluded*
from it by that same split.

The range is four orders of magnitude — 0.01% to 154% — which is the single most important
thing in this document. **There is no package-wide answer to "how much of a fit is
post-nuisance work", and any recommendation that does not say which flavour and which
workload it is about is not a recommendation.**

Three things follow.

- **`library="glm"` is the most favourable denominator there is.** A `default` fit costs
  ≈37× more per row (`docs/roadmap.md`), so the Newton column above is ≈0.14% of a
  realistic fit and the one-step column ≈0.8%. On a single fit with a real learner library,
  none of this is visible — which is the roadmap's existing conclusion, unchanged.
- **The repeated workloads invert it, and they are the whole case for adopting anything.** A
  25-point truncation sweep is **150% of a `glm` fit** because it retargets 25 times against
  one set of nuisances; that ratio does *not* shrink by 37× under a real library, because
  the sweep's numerator is post-nuisance work all the way down. This is also exactly where
  §4's compilation cost amortises.
- **DR-TMLE's `retarget` costing 1.5× its own fit is not an arithmetic problem.** §6 is.

So the honest summary is:

> On a single fit with a realistic learner library, none of this is visible. On a repeated
> post-nuisance workload — a sensitivity sweep, a simulation study, a bootstrap, a survival
> curve over many horizons — the compiled kernels are the difference between minutes and
> seconds, and the multiplier bootstrap's memory behaviour is the difference between running
> and not running at several million rows.

---

## 6. The finding that is not about numba

`cleverly.learners.thread_limit` is entered once per learner fit and costs **1.44 ms** per
entry, because `threadpoolctl.threadpool_limits` constructs a `ThreadpoolController` that
walks every shared object the process has loaded (`dl_iterate_phdr`). Measured:

- **DR-TMLE `retarget`**: 41 s of 72 s inside `threadpoolctl` — 57%, and the reason a
  `retarget` costs 2.2× the fit it is supposed to be a cheap re-run of.
- **LTMLE fit**: 0.84 s of 2.07 s — 40%, against 0.043 s in the actual loss gradients.

No compiled kernel addresses this and no parallel axis helps. The fix is to build the
controller once rather than per fit.

**Since done**, in its own change with its own tests, as this section said it should be:
0.759 ms per entry against 0.0129 ms reusing one — **59×** — and through the API a DR-TMLE
`retarget` 5.206 s → 2.674 s, **49% of it removed**. The 57% and 40% above are cProfile
figures and were overstated by the profiler's per-call overhead, which falls hardest on the
code making the most calls; the wall-clock numbers are in
[`thread_limit_profile.md`](thread_limit_profile.md), along with the one real hazard a
cached controller has (a pool loaded later — LightGBM is imported lazily) and how it is
handled. `benchmarks/numba/kernels/drtmle.py` carries `thread_limit_overhead()` so the
before-number is still one call away.

---

## 7. Recommendation

**As issued, and then what acting on it found.** Steps 1 and 2 were right and are done;
steps 3 and 4 were the ones that did not survive being built, for a reason worth keeping on
the record rather than editing away.

1. **Fix `thread_limit`** (§6). Not numba, largest effect, cheapest change. ✅ **Done** —
   59× per entry, 49% of a DR-TMLE `retarget`
   ([`thread_limit_profile.md`](thread_limit_profile.md)).
2. **Fix the LTMLE mask construction** (§2.3) and **defer the one-step arm updates**
   (§2.5). Pure numpy, no compilation. ✅ **Mask fix done** — `O(T²n)` → `O(Tn)`, 8.7× on
   the mask term at `T = 20` and **0.06% of a fit**
   ([`longitudinal_masks.md`](longitudinal_masks.md)). The one-step decision is still open
   and still a decision, not a refactor.

**Then, if `numba` is to become a runtime dependency** — and the answer, after building the
numpy side of both candidates, is **not yet**:

3. ~~**`cluster_sums`** and the **multiplier bootstrap** (§2.1–2.2)~~. Both were rewritten
   in numpy first, as the plan's own rule required, and both ratios collapsed. The
   bootstrap's numpy path is now 3.4–3.9× faster than the one measured here and allocates a
   a 32 MB buffer budget — so the capability change is had without the dependency, and a
   compiled kernel now has to beat *that*. `cluster_sums` against the codes the container
   actually produces is 1.02× at five estimands and 0.74× at a million rows; its 5.5–10.2×
   was mostly an `np.unique` the package no longer runs.
   ([`bootstrap_numpy.md`](bootstrap_numpy.md), [`cluster_integration.md`](cluster_integration.md))
4. **The LTMLE and survival recursions** (§2.3–2.4) and **the CV-TMLE fold loop** (§2.6)
   remain the largest *compiled* opportunities, and the denominator is now measured rather
   than inferred: the whole backward recursion's package-owned arithmetic — fluctuations,
   clever covariates, masks, pseudo-outcomes — is **1.5% of a `glm` longitudinal fit**,
   where `inference` is 20%. A 10× on 1.5% is not a dependency.

**Retain numpy** for the targeting Newton, the fused influence curves, the DR-TMLE
alternation arithmetic, and — added by the work above — cluster aggregation, whose
kernel-level win is real from about seven estimands and whose *fit-level* share is not worth
a dependency: 2.3–2.8× on `influence_covariance`, ~1.1× on a clustered `retarget` at
`n = 10⁵`, and nothing at `n = 20,000`. The crossover in `m` itself is **unlocated** — see
[`cluster_integration.md`](cluster_integration.md), which named twenty and should not have.

**Defer** the CTMLE candidate scoring until candidates are cheap enough for it to be
visible, and the MSM Gram until something else makes the projection matter.

**Do not** adopt numba wholesale, and the case for adopting it at all is now weaker than
this document concluded. Three of the twelve kernels do not clear the continuation bar at
all; the obvious task-parallel arm (threads over numpy fold bodies) *regresses* past two
cores; the single largest package-owned cost in two flavours was a context manager, and is
gone. What is left, after the numpy work the two clearest "wins" turned out to be, is one
open question rather than a plan: **can a compiled kernel consume numpy's packed multiplier
bytes and beat the blocked expansion?** Nothing else in this suite currently clears a bar
that is worth a runtime dependency.

---

## 8. What this run does not answer

Stated so a reader does not mistake a gap for a covered case.

- **Hybrid `w × t` splits are not measured.** The mode exists in
  `implementations/numpy_reference.py` and resolves a plan, but no kernel here accepts a
  worker-count *and* a thread-count: the compiled kernels take one `prange` axis and the
  thread-pool arm takes one worker count. The one kernel with a genuine two-level
  structure is `cvtmle_fold_targeting` (folds × rows), and measuring it properly means
  giving it a nested form rather than a flag. On four cores the only splits available are
  2×2 and the two degenerate ones, which is not enough to see a trend, so it was left for
  a machine where it could be answered.
- **Process-level parallelism is not measured**, only threads. For the fold and regimen
  axes a process pool would have to pickle a slice of every array per task, and on this box
  that transfer is larger than the task; on a machine where it is not, the GIL result in
  §2.6 is the reason to try it.
- **The nonparametric bootstrap is not benchmarked, and that is a scope decision rather
  than an omission.** `cleverly.inference.run_bootstrap` refits the whole estimator on each
  resample, so it is nuisance-fitting-bound by construction — the plan's own rule puts
  learner fits out of scope, and a compiled kernel cannot reach inside a LightGBM fit. What
  *is* covered is the two package-owned halves that a cluster bootstrap composes:
  `cluster_sums` for the aggregation and `multiplier_bootstrap` for the resampling, and a
  clustered multiplier bootstrap is the first applied to the influence curves followed by
  the second applied to the result. Neither the composition nor its index generation has a
  kernel of its own, because neither adds arithmetic to what those two already measure.
- **Core counts stop at four**, because the box has four. Several kernels are still above
  0.7 efficiency there and their curves have not turned over. **And nothing in CI reaches
  even that**: both jobs in `.github/workflows/numba-benchmark.yml` ran `--num-cores 1 2`,
  and the `full` one passed it *alongside* `full.yaml`'s own `[1, 2, 4, 8]`, which the flag
  replaces rather than narrows — so the config's 4 and 8 were discarded and no run anywhere
  has produced an eight-core point. The flag is gone from that job and the core counts above
  a runner's are now recorded as skipped rows rather than silently dropped, but the *gap* is
  unchanged until someone dispatches it at a larger `runner:`. So: **nothing here says numba
  would not help a large repeated workload on 8–32 physical cores.** It says the measured
  serial and low-core workloads do not justify it.
- **Every number was taken in randomised block order**, not in the rotation the harness now
  uses — see the provenance note in §1. Ratios of 3× and above are not in doubt; ratios
  within a few percent of 1.0, such as `cluster_sums`'s 1.02× at five estimands, are not
  resolved by this run and would need a paired rerun on the current harness.
- **`cluster_sums` has a reproducible discontinuity between five and seven estimands** that
  nothing here explains, and the harness grid (`n_estimands: [1, 5, 20]`) has no point
  between 7 and 20. So its crossover in `m` is unlocated rather than known; see
  [`cluster_integration.md`](cluster_integration.md).
- **`n` stops at 10⁶** for the row-indexed kernels and lower for the recursions, which is
  where a `numpy` reference at `T = 20` reaches four seconds a call. Nothing here is
  extrapolated past what was run.
- **The `default` learner preset is measured for the pipelines only**, not for every
  scenario, because a `default` fit is tens of seconds and the pipeline table needs one per
  flavour. The 37×-per-row ratio the shares are scaled by is the roadmap's, re-measured
  where it was cheap to.
- **There is no coverage or bias comparison between the numpy and compiled paths, and
  there could not be one yet.** These kernels are benchmark code: nothing under `src/`
  calls them, so a simulation study of "the compiled estimator" would be a study of an
  estimator that does not exist. What *is* checked is the layer below that, which is what
  makes such a study unnecessary until adoption: the compiled implementations agree with
  the shipped path to a stated tolerance on every kernel, agree on the algorithmic facts a
  float comparison cannot see (iteration count, convergence flag, selected candidate), and
  do not move with the thread count. When a kernel is adopted into `src/`, the statistical
  tier it lands in is `pytest -m slow`, and the comparison to run there is coverage against
  the *estimand*, not against the numpy path — because by then the numpy path is gone.

## 9. What would change these numbers

- **A machine with more cores.** Efficiency is measured to four here; several kernels are
  still above 0.7 at four threads and would be worth re-measuring at 16.
- **HAL.** The roadmap's trigger for native code is unchanged by any of this: a nuisance
  learner that is not scikit-learn moves the whole denominator, and everything in §4 is
  computed against learners that are.
- **`library="glm"` at several million rows**, which the roadmap already flags as the one
  production configuration where `solve_one_step` dominates a fit. That is exactly §2.5's
  kernel, and it is the case where its 2.8×–5.7× would be a fit-level number rather than a
  kernel-level one.
