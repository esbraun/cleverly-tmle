# Findings: numba and parallelism after the nuisances are fitted

The recommendation half of the investigation whose profile is
[`candidate_inventory.md`](candidate_inventory.md). Read that first — it is what the work
was sized against, and three of the things it is natural to expect turn out to be false.

Everything here is **post-nuisance**: learner fits are outside every timed region. That is
what makes `n = 1,000,000` a couple of seconds and the scaling questions answerable by
direct measurement — and it is also why every speed-up below has to be multiplied by §4's
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
favourable* denominator available (see §4).

| kernel | serial | parallel (4 cores) | memory | decision |
| --- | ---: | ---: | --- | --- |
| `multiplier_bootstrap` | **2.4–2.5×** | **7.4–7.6×** | never forms the `(chunk, n)` array | **adopt numba parallel** |
| `cluster_sums` | **5.4–10.6×** | **9.3–20.6×** | unchanged | **adopt numba** (parallel above ~10⁵ rows) |
| `ltmle_backward_recursion` | **2.1–4.8×** | **8.2–16.1×** | unchanged | **adopt numba parallel** — *after* the mask fix |
| `survival_incidence` | **3.7×** | **10.2×** | unchanged | **adopt numba parallel** |
| `one_step_walk` | **2.8×** | **4.1–5.7×** | fewer temporaries | **fix the algorithm, then adopt** |
| `ctmle_candidate_scores` | 2.6× | 9.7× | unchanged | **defer** — 11 ms of a 190 s fit |
| `msm_gram` *(control)* | **3.1–3.4×** | n/a | unchanged | **control that turned positive; see §3** |
| `fused_influence_curves` | 1.1–2.8× | 1.6–4.1× | unchanged | **retain numpy** — the share is a tenth of a tenth |
| `drtmle_reduction_rounds` | 0.93–1.04× | 1.6–1.9× | unchanged | **retain numpy** |
| `newton_targeting` *(control)* | 1.2–1.4× | n/a | unchanged | **retain numpy**, as expected |
| `cvtmle_fold_targeting` | see §2.6 | see §2.6 | unchanged | see §2.6 |

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

**The memory argument is separate and may matter more than the speed.** The roadmap names
this array as one of two allocations that break before any arithmetic does — ≈9.5 GB at
`n = 5,000,000`. The fused kernel's working set is `block × m` doubles: **32 KB, at any
`n`.**

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

### 2.6 CV-TMLE's fold loop — the task-parallel axis

`TMLE._solve_by_fold` is a serial Python `for` over folds that are independent by
construction: a fold's `epsilon` is fitted only against rows whose nuisance predictions came
from models that never saw them. Four arms measure the two parallelisms against each other —
the fold loop over a thread pool, and `prange` over folds — because folds cut each task's
rows by `1/F` while giving the task axis more tasks, and where those cross is a property of
the machine rather than of the algorithm.

See `latest/summary.md` for the table at 2, 10 and 20 folds. The headline is that the
**per-fold body is the Newton solve**, which §3 shows is the one kernel here that
compilation does not help — so this axis is about *scheduling*, and its ceiling is the
serial stitch that every arm still has to do.

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
DR-TMLE `retarget`'s time goes anyway: §5 is.

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

## 3. The negative controls, one of which was not

`newton_targeting` behaves exactly as `benchmarks/bench_tmle.py` said it would: **1.4× at
`n = 10⁴`, 1.18× at `n = 10⁵`**, falling with size and below the 1.25× continuation bar at
the larger one. Its inner work is `x @ eps`, `x.T @ (w r)` and `x.T @ (x v)` — BLAS calls —
plus one vectorised `expit`. A compiler pays where the interpreter is in the inner loop, and
here it is not. **Retain numpy**, confirmed on this harness rather than quoted from another.

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

## 4. The denominator: what any of this is worth

`benchmarks/results/latest/pipelines.md` carries the measured shares. The shape of the
answer:

- A `glm` fit's post-nuisance half is **2–6%** of it under Newton targeting and **31%**
  under the one-step walk.
- A `default` fit costs **≈37× more per row**, so the same shares are **0.06–0.17%** and
  **0.8%**.
- The exceptions are the *repeated* workloads, and they are the whole case for adopting
  anything: a 25-point truncation sweep is **97% of a `glm` fit** because it retargets 25
  times against one set of nuisances; a DR-TMLE `retarget` is **220%** of its own fit.

So the honest summary is:

> On a single fit with a realistic learner library, none of this is visible. On a repeated
> post-nuisance workload — a sensitivity sweep, a simulation study, a bootstrap, a survival
> curve over many horizons — the compiled kernels are the difference between minutes and
> seconds, and the multiplier bootstrap's memory behaviour is the difference between running
> and not running at several million rows.

---

## 5. The finding that is not about numba

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

## 6. Recommendation

**Adopt, in this order**, each behind the `bench` extra becoming a runtime dependency —
which is a real cost and the reason the order is by value and not by ratio:

1. **Fix `thread_limit`** (§5). Not numba, largest effect, cheapest change.
2. **Fix the LTMLE mask construction** (§2.3) and **defer the one-step arm updates**
   (§2.5). Pure numpy, 1.7–2.4× and 1.5–1.7×, no new dependency, no compilation.
3. **`cluster_sums`** and the **multiplier bootstrap** (§2.1–2.2) as compiled kernels.
   These are the two with both a large ratio and a real share, and the bootstrap's memory
   behaviour is a capability change rather than a speed-up.
4. **The LTMLE and survival recursions** (§2.3–2.4), parallel over regimens and horizons.
   The largest absolute savings in the package, and only after (2).

**Retain numpy** for the targeting Newton, the fused influence curves, and the DR-TMLE
alternation arithmetic.

**Defer** the CTMLE candidate scoring until candidates are cheap enough for it to be
visible, and the MSM Gram until something else makes the projection matter.

**Do not** adopt numba wholesale. Six of the eleven kernels here do not clear the plan's
continuation bar, two of the wins are numpy rewrites rather than compilations, and the
single largest package-owned cost in two flavours is a context manager.

---

## 7. What would change these numbers

- **A machine with more cores.** Efficiency is measured to four here; several kernels are
  still above 0.7 at four threads and would be worth re-measuring at 16.
- **HAL.** The roadmap's trigger for native code is unchanged by any of this: a nuisance
  learner that is not scikit-learn moves the whole denominator, and everything in §4 is
  computed against learners that are.
- **`library="glm"` at several million rows**, which the roadmap already flags as the one
  production configuration where `solve_one_step` dominates a fit. That is exactly §2.5's
  kernel, and it is the case where its 2.8×–5.7× would be a fit-level number rather than a
  kernel-level one.
