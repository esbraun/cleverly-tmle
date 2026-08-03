# Findings: numba and parallelism after the nuisances are fitted

The recommendation half of the investigation whose profile is
[`candidate_inventory.md`](candidate_inventory.md). Read that first — it is what the work
was sized against, and three of the things it is natural to expect turn out to be false.

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

---

## 1. The answer in one table

Speed-ups are against the shipped numpy path at one core, on the same input object.
"Share" is the kernel's weight inside a `library="glm"` post-nuisance step — the *most
favourable* denominator available (see §5).

| kernel | serial | parallel (4 cores) | memory | decision |
| --- | ---: | ---: | --- | --- |
| `multiplier_bootstrap` | **2.4–2.5×** | **7.4–7.6×** | **427.6 MB → 1.6 MB per call** | **adopt numba parallel** |
| `cluster_sums` | **5.4–10.6×** | **9.3–20.6×** | unchanged | **adopt numba** (parallel above ~10⁵ rows) |
| `ltmle_backward_recursion` | **2.1–4.8×** | **8.2–16.1×** | unchanged | **adopt numba parallel** — *after* the mask fix |
| `survival_incidence` | **3.7×** | **10.2×** | unchanged | **adopt numba parallel** |
| `one_step_walk` | **2.8×** | **4.1–5.7×** | fewer temporaries | **fix the algorithm, then adopt** |
| `ctmle_candidate_scores` | 2.6× | 9.7× | unchanged | **defer** — 11 ms of a 190 s fit |
| `msm_gram` *(control)* | **3.1–3.4×** | n/a | unchanged | **control that turned positive; see §3** |
| `fused_influence_curves` | 1.1–2.8× | 1.6–4.1× | unchanged | **retain numpy** — the share is a tenth of a tenth |
| `drtmle_reduction_rounds` | 0.93–1.04× | 1.6–1.9× | unchanged | **retain numpy** |
| `newton_targeting` *(control)* | 1.2–1.4× | n/a | unchanged | **retain numpy**, as expected |
| `cvtmle_fold_targeting` | 1.5–4.1× | **3.2–14.3×** | unchanged | **adopt numba parallel over folds** |

And the one that is not a compilation question at all:

| finding | magnitude | decision |
| --- | --- | --- |
| `cleverly.learners.thread_limit` builds a fresh `ThreadpoolController` per learner fit | **1.44 ms per entry**; 57% of a DR-TMLE `retarget`, 40% of an LTMLE fit | **fix in numpy-land**, in its own change |

---

## 2. Kernel by kernel

### 2.1 The Rademacher multiplier bootstrap — the clearest adoption

`docs/roadmap.md` already records that this is 92–95% multiplier *generation*, and the
package already took the cheap half of that (packing bits instead of drawing float64
uniforms). What it did not change is the shape: a `(chunk, n)` float64 array is
materialised in order to be consumed once, and then multiplied into `centred` by a BLAS
`dgemm` whose left operand is a matrix of plus and minus ones.

A fused kernel never forms it. Each row draws its bit and is added to or subtracted from an
`m`-vector accumulator, so both the array and the multiply disappear together.

| n | numpy | numba | numba, 4 threads |
| ---: | ---: | ---: | ---: |
| 10,000 | 53.1 ms | 21.0 ms (2.5×) | 7.2 ms (7.4×) |
| 100,000 | 448.6 ms | 184.6 ms (2.4×) | 59.0 ms (7.6×) |

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
every later one reads a delta of zero. The numbers above are per-call Python-level
allocation, taken in an untimed pass.)

**Reproducibility is by construction, not by luck.** The draw is a splitmix64 hash of
`(seed, replicate index)`, so a replicate's multipliers do not depend on which thread ran
it or how many threads there were. One thread and four threads give bit-identical answers,
which `tests/unit/test_numba_benchmark.py` pins — and the same test checks the fused
statistic against signs reconstructed in numpy from the same counter, because the
cross-generator quantile gate is a Monte Carlo one and would pass a kernel that dropped a
column and got lucky.

### 2.2 `cluster_sums` — the largest ratio in the suite, mostly from serial numba

Two costs, and only one of them is about compilation. `np.unique` **sorts** to densify
labels that only need hashing, and then one `np.bincount` per estimand re-reads the same
index vector `m` times.

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
controller once rather than per fit. It is deliberately **not** done here: changing when
thread limits are applied is a change to the library's runtime behaviour with its own
correctness surface (`set_thread_limit(None)`, nested fits, joblib workers), and it belongs
in its own change with its own tests. `benchmarks/numba/kernels/drtmle.py` carries
`thread_limit_overhead()` so the number is one call away.

---

## 7. Recommendation

**The first two steps need no new dependency and should happen regardless of what is
decided about numba.**

1. **Fix `thread_limit`** (§6). Not numba, largest effect, cheapest change: 57% of a
   DR-TMLE `retarget` and 40% of an LTMLE fit.
2. **Fix the LTMLE mask construction** (§2.3) and **defer the one-step arm updates**
   (§2.5). Pure numpy, 1.7–2.4× and 1.5–1.7×, no compilation. The second needs a decision
   about which estimator is intended when the shrink bound binds, so it is a change with a
   question in it rather than a pure refactor.

**Then, if `numba` is to become a runtime dependency** — a real cost, and the reason the
order below is by value rather than by ratio:

3. **`cluster_sums`** and the **multiplier bootstrap** (§2.1–2.2). The two with both a
   large ratio and a real share, and the bootstrap's memory behaviour is a capability
   change rather than a speed-up: 32 KB of working set at any `n`, against an allocation
   the roadmap already names as one of the two that break first at scale.
4. **The LTMLE and survival recursions** (§2.3–2.4), parallel over regimens and horizons,
   and **the CV-TMLE fold loop** (§2.6) parallel over folds. The largest absolute savings
   in the package — and only after (2), or the compiled kernel is a fast version of
   redundant work.

**Retain numpy** for the targeting Newton, the fused influence curves, and the DR-TMLE
alternation arithmetic.

**Defer** the CTMLE candidate scoring until candidates are cheap enough for it to be
visible, and the MSM Gram until something else makes the projection matter.

**Do not** adopt numba wholesale. Three of the twelve kernels here do not clear the plan's
continuation bar at all, two of the clearest wins are numpy rewrites rather than
compilations, the obvious task-parallel arm (threads over numpy fold bodies) *regresses*
past two cores, and the single largest package-owned cost in two flavours is a context
manager. The plan's own expected shape — "not numba everywhere or numba nowhere" — is what
the measurement returned.

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
- **Core counts stop at four**, because the box has four. Several kernels are still above
  0.7 efficiency there and their curves have not turned over.
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
