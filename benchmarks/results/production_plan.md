# Production plan: what to do with the numba investigation

> ## Outcome — the plan was executed; Steps 0–5 and 7 are done, Step 6 is not
>
> | step | outcome |
> | --- | --- |
> | 1. `thread_limit` | ✅ **59× per entry**, 49% of a DR-TMLE `retarget` removed, 1.34× an LTMLE fit. [`thread_limit_profile.md`](thread_limit_profile.md) |
> | 2. multiplier bootstrap in numpy | ✅ **3.4–3.9×**, allocation from 1,881 MB to 92 MB at `n = 10⁶`, seeded stream bit-identical. float32 measured (7.2×) and deliberately not taken. [`bootstrap_numpy.md`](bootstrap_numpy.md) |
> | 3. cluster densification | ✅ **1.24–13.8×** on the kernel, 2.3–2.8× on `influence_covariance`, ~1.1× on a clustered `retarget` at `n = 10⁵` and nothing at `n = 20,000`. [`cluster_integration.md`](cluster_integration.md) |
> | 4. phase timing | ✅ `LTMLE.profile_phases()`; the learners are **76%** of a `glm` longitudinal fit, `inference` 20%, the whole backward recursion 1.5%. |
> | 5. prefix masks | ✅ `O(T²n)` → `O(Tn)`, **8.7× on the mask term at `T = 20`** and **0.06% of a fit**. [`longitudinal_masks.md`](longitudinal_masks.md) |
> | 7. reprofile | ✅ DR-TMLE's `retarget` now costs 1.01× its fit; the largest arithmetic line is 2.9% and `threadpoolctl` is gone from the profile. A measured negative, not a deferral. |
> | 6. `src/cleverly/_kernels/` | ❌ **not built, on purpose.** It was conditional on a kernel clearing Steps 2–3, and none does: the bootstrap's numpy path is now faster than the compiled kernel measured against the old one, and cluster aggregation is 1.02× at five estimands and 0.74× at a million rows. Building the seam now would be an abstraction with nothing to put in it. |
>
> **The through-line.** Every one of the three "adopt numba" recommendations this plan was
> written to check turned out to be a numpy result: an expansion that did not need doing, a
> sort that had already been done elsewhere, and a mask rebuild that was quadratic. The
> largest single win in the whole investigation — 49% of a `retarget` — was a context
> manager. `numba` remains a benchmark-only dependency and the open question is now exactly
> one: §1.2's.


The third document of the sequence. [`candidate_inventory.md`](candidate_inventory.md) is the
profile, [`findings.md`](findings.md) is the measurement, and this is what to *build* — which
is a different question, because a kernel that wins in a benchmark harness wins against
whatever baseline that harness happened to ship, and the baseline is the part nobody
re-derives.

This plan is a revision of a proposed one. Most of the proposal is accepted and reordered;
**four of its load-bearing claims are wrong**, and each is contradicted below by a
measurement rather than by an argument. The largest of them inverts the proposal's own
headline: the multiplier bootstrap is presented as the strongest numba candidate in the
package, and a pure-numpy rewrite of it beats the compiled parallel kernel on one core.

> **Provenance for every number in §1.** Taken while writing this, on the four-core Intel
> Xeon @ 2.80 GHz container these sessions run in, `/proc/loadavg` under 0.6, Python 3.11,
> numpy 2.4.6, numba 0.66.0, threadpoolctl 3.6.0. Timings are medians of interleaved
> repetitions. This is the same box `findings.md` was measured on, which is what makes the
> comparisons below comparisons rather than anecdotes.

---

## 1. Four claims that do not survive a measurement

### 1.1 The multiplier bootstrap's cost is not the random draw, and numpy can take most of it

`candidate_inventory.md` and `docs/roadmap.md` both record the rademacher path as "92–95%
multiplier *generation*", and both point the reader at the draw. Split the generation at
`n = 100,000`, `m = 7`, one 256-replicate chunk:

| step | ms | share of the chunk |
| --- | ---: | ---: |
| `rng.integers(0, 256, (256, n/8), uint8)` — **the seeded draw** | **3.5** | 2% |
| `np.unpackbits(...)` — bits to `uint8` | 1.7 | 1% |
| `_SIGNS[bits]` — **`uint8` to a 205 MB float64 matrix** | **159.0** | **89%** |
| `xi @ centred` — the `dgemm` | 12.6 | 7% |

The cost is not drawing the bits. It is expanding one bit per element into a float64 and
writing 205 MB, in numpy, in `src/`, today. Two consequences, and the second is the one that
changes the plan.

**A blocked numpy rewrite is most of the win, with no dependency.** Through the production
arithmetic at `n = 100,000`, `m = 7`, `B = 1,000`, interleaved, three repetitions each:

| implementation | seconds | speed-up | critical value |
| --- | ---: | ---: | --- |
| shipped (`_CHUNK = 256`, `_SIGNS[...]`) | 0.979 | 1.00× | 2.645745 |
| blocked float64 (`chunk = 64`, in-place `x = b.astype(f8); x *= 2; x -= 1`) | 0.512 | **1.91×** | 2.645745 — *identical* |
| blocked float32 (same, `float32` expansion and `dgemm`) | **0.136** | **7.18×** | 2.645744 |

`findings.md` §2.1 puts the compiled kernel at **2.4–2.5× serial and 7.4–7.6× on four
cores**. So the float64 blocked rewrite matches the compiled kernel's serial arm and returns
the *bit-identical* critical value, and the float32 rewrite matches the compiled kernel's
four-core arm **on one core**, with no runtime dependency, no 2.0 s compilation, and no
change to the seeded stream. Memory falls with the block: `chunk = 64` at float32 is 25.6 MB
against 204.8 MB, and `chunk = 16` is 6.4 MB.

The proposal anticipated this in the abstract — its §3.2 says "Numba must beat the best
reasonable NumPy implementation, not only the shipped implementation" — and then classified
the kernel as a **strong production candidate** anyway, on ratios measured against the
shipped shape. The correct classification is **fix the numpy path; re-open the numba
question against the fixed one**.

The float32 arm is a numerical change and is not free: it is a change to the arithmetic of a
Monte Carlo quantile, and it needs its own justification (the resampling error at `B = 1,000`
is ~1e-2 on the critical value; the float32 accumulation error measured 1e-6). The float64
arm needs none — it is the same values in a different order of operations, and it measured
identical.

### 1.2 The RNG-compatibility trade-off is an artefact of the benchmark kernel

The proposal's §3.4 asks for a policy decision — preserve the seeded stream, or ship the new
generator as experimental and document that the stream differs. Neither is necessary. The
incompatibility exists because `benchmarks/numba/kernels/bootstrap.py` draws its own signs
from a splitmix64 hash of `(seed, replicate)`, which was the right choice *for a benchmark*
(it makes the result independent of the thread count by construction) and is not forced on a
production kernel.

The draw is 2% of the cost (§1.1). Keep `rng.integers` exactly where it is, hand the compiled
kernel the **packed bytes**, and the stream is preserved by construction — the 89% that is
worth removing is the expansion, not the draw. Also measured: the chunk size does not affect
the seeded output either (`_CHUNK = 256` and `chunk = 64` returned identical critical values
at both sizes tried), so the block size is free to tune. Pin that in a test rather than
relying on it.

One caution against assuming the compiled version then wins. A packed-byte kernel written the
obvious way — extract the bit in the inner loop — measured **275 ms serial and 69 ms on four
threads** per 256 replicates against the improved numpy's 26–31 ms, because the shift-and-mask
defeats the vectoriser that the `astype`-then-`dgemm` path gets for free. If a compiled kernel
is written here it must unpack a byte's eight signs once into registers, and it must be
measured against §1.1's baseline and not against the shipped one.

### 1.3 `tracemalloc` does see numba's allocations — and the harness's docstring says otherwise

The proposal's §5 replaces the memory instrument wholesale, on the premise that `tracemalloc`
"may not observe native allocations made inside compiled kernels". The premise is false on
this stack, and `benchmarks/numba/timing.py`'s own docstring asserts the same false thing:

> It does *not* see allocations a compiled kernel makes on numba's own side of the boundary,
> so a `prange` kernel's per-thread scratch is invisible here

Two measurements:

- 80 MB allocated by two `np.empty`/`np.zeros` calls **inside** an `@njit` function reads
  `tracemalloc` peak **80,000,224 bytes** — the same 80 MB the identical numpy allocations read.
- `numba_cluster_sums_threadlocal` at `n = 200,000`, `C = 50,000`, `m = 20`, four threads reads
  **40.0 MB**, against the serial kernel's 9.99 MB. The difference is 30 MB against an expected
  `threads × C × m × 8 = 32 MB` thread-local block. The per-thread scratch the docstring calls
  invisible is *exactly* what the number shows.

So: **correct the docstring and the paragraph in `findings.md` §3.2 that rests on it, and keep
`tracemalloc` as the primary instrument.** It is per-call, it is comparable across
implementations, and it is the reason `findings.md` could reject peak RSS (a process
high-water mark reads zero for every implementation after the first).

There is still a real gap, narrower than the proposal's: `tracemalloc` traces CPython's three
allocator domains, so it sees numpy and numba's NRT and does **not** see a library that calls
`malloc` directly — OpenBLAS scratch is the case that matters here — and RSS is what decides
whether a job is killed. Add an isolated-process incremental-RSS arm **for the two adoption
candidates at one configuration each**, as a confirmation that nothing is hiding below the
allocator, and report it beside the per-call column rather than in place of it. Do not rebuild
the schema around it.

### 1.4 Total speed-up and parallel scaling are already separated — in the code, not in the prose

The proposal's §6 asks for a schema change. `benchmarks/numba/reporting.py::_scaling_table`
already computes the speed-up against *that implementation at one core* and the generated
report already says so in as many words:

> Speed-up is against *that* implementation at one core, so it isolates what the added cores
> bought; the verdict table above compares against the numpy reference instead.

What conflates the two is `findings.md` §1, whose column heading reads `parallel (4 cores)`
and whose contents are total speed-ups against numpy-at-one-core. That is a prose defect in
one table, and the fix is to widen that table to the triple the proposal specifies — total
vs numpy-1, parallel vs same-backend-1, efficiency — which the harness can already supply.
Two smaller corrections in the same place: `_scaling_table`'s `efficiency` column divides by
`num_cores_requested` while its speed-up column is same-backend, so that column is already
right, and the regression flag (`> 1.05 ×`) exists and is emitted. Do not rewrite what works.

Two of the proposal's other sections are in the same position and should be rescoped the same
way rather than started from scratch:

- **§7, task parallelism.** The `numpy_threads` arm exists and *executes*: `findings.md` §2.6
  reports the unchanged numpy fold bodies over a `ThreadPoolExecutor` reaching 1.46× on two
  cores and regressing to 1.18× on four. What is genuinely absent is **process** workers and
  the hybrid `w × t` split, both of which `findings.md` §8 already names as unanswered and
  gives the reason for (on four cores the only non-degenerate split is 2×2). This is finishing
  a stated gap, not adding a missing axis.
- **§13, CI.** All three tiers exist. Correctness runs on every pull request
  (`.github/workflows/ci.yml::numba-correctness`); the smoke tier is scheduled weekly; the full
  sweep is dispatch-only, and `numba-benchmark.yml`'s own header already says a two-core hosted
  runner's scaling table is worth reading for its shape and not its numbers. What is missing is
  a fast-tier test of the *production* backends, which cannot exist until they do, and a
  controlled-hardware tier, which needs a named machine rather than another workflow file.

---

## 2. What the proposal gets right, unchanged

Recorded so the revision is not read as a rejection:

- **The framing.** "No benchmark-only kernel is described as adopted production architecture"
  is correct and `findings.md` §1's `decision` column does not meet it.
- **The kernel/production split** (§2). Validation, dtype handling, diagnostics, result objects
  and public random-state behaviour stay in Python; backends take validated primitive arrays.
- **"Benchmark code calls the production backend rather than maintaining a separate
  implementation."** This is the single most valuable structural item in the proposal, and it
  is what would have caught §1.1 a month ago: the benchmark's numpy reference reproduces the
  shipped *shape*, so a numpy improvement to the shipped path is invisible to it by
  construction.
- **The longitudinal mask fix** (§10) — confirmed below.
- **The threadpoolctl fix** (§8) — confirmed below, with a number.
- **Explicit phase timing** (§11), and not inferring phase shares by matching filenames in a
  cProfile.
- **The whole defer list** (§12), for the reasons given there.

---

## 3. The two remaining findings, re-measured

### 3.1 `thread_limit`: 53× per entry, and the hazard is a lazy import

`cleverly.learners._threads.thread_limit` is entered at six sites in `_fitting.py`, once per
learner fit and once per predict. Measured here, in a process with OpenBLAS and OpenMP loaded
but *not* LightGBM:

| | ms per entry |
| --- | ---: |
| `with thread_limit(1)` (the shipped path) | **0.688** |
| `ThreadpoolController()` alone | 0.608 |
| `with cached_controller.limit(limits=1)` | **0.013** |

**53×**, and the 0.688 ms here against `findings.md`'s 1.44 ms is the difference the loaded
shared objects make — the walk is over `dl_iterate_phdr`, so the cost scales with what the
process has imported, and a real fit has imported more.

The correctness surface the proposal lists is right, and one item on it is the whole design
question: **libraries loaded after the controller is built**. `src/cleverly/learners/library.py`
imports LightGBM *lazily*, inside the function that builds the learner, so a controller cached
at first `thread_limit` entry can predate the OpenMP pool it is supposed to limit. threadpoolctl
3.6 has no refresh — a rebuild is a new `ThreadpoolController()` — so the fix should be a cached
controller plus an explicit invalidation at the one place in the package that imports a backend,
not a polling heuristic. A public escape hatch for a user who `dlopen`s something themselves
costs one function.

### 3.2 The longitudinal masks are `O(T² n)`, confirmed in the source

`LongitudinalData.at_risk(t)` and `.following(t)` each call:

- `_through(uncensored, t)` = `uncensored[:, :t].all(axis=1)` — **`O(t·n)`**;
- `followed_through(assignment, t)` — an explicit Python loop of `t` boolean `&` passes, plus an
  `assignment_matrix` call **per invocation**;
- `_event_free(event, t)` — a single column read, `O(n)`, already right, and worth saying so:
  the `event` matrix is stored cumulatively for exactly this reason, and the fix for the other
  two is the same idea applied to them.

Summed over the nodes that is `Σ_t O(t·n) = O(T² n)` per regimen, and a survival fit runs a pass
per horizon on top. `findings.md` §2.3 measures the prefix-mask arm at 1.00× at `T = 5` and
**1.71–2.44× at `T = 20`**.

So the fix is real, and its *production* value at the `T` anyone runs is unmeasured — the
package's own fixtures use `T = 2`, where the term is invisible. That is why §4 orders the phase
timing before the mask fix rather than after it, which reverses the proposal's order.

### 3.3 Cluster aggregation: the numpy baseline to beat is a densify-once, not a `reduceat`

The proposal's §4.2 asks whether labels could be densified once and reused across estimates,
covariance, and bootstrap draws. They can — `cluster` is already a validated array on the
container — and the answer is worth more than the proposal assumes:

| n | C | m | shipped `cluster_sums` | `np.unique` alone | pre-densified `bincount`s | gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 1,000 | 5 | 5.26 ms | 2.77 ms | 1.75 ms | **2.72×** |
| 100,000 | 1,000 | 20 | 13.70 ms | 2.89 ms | 10.85 ms | 1.31× |
| 1,000,000 | 10,000 | 5 | 64.38 ms | 44.18 ms | 17.71 ms | **3.54×** |

The sort is most of the cost at a small estimand count and a large `n`, and it is paid on every
call — including once per bootstrap draw. `findings.md` §2.2 already records that the *other*
numpy arm fails (one sort plus `np.add.reduceat` is 0.39–0.88× of the shipped path), and it
draws the right conclusion from it — "what is wanted is not a better sort, it is not sorting" —
but the arm that does not sort **at all**, because the codes were densified upstream, was not
run.

This does not eliminate the numba case the way §1.1 does. What remains after the densify is the
`m`-passes-over-one-index-vector cost, which is what the single fused pass removes, and at
`m = 20` that is the dominant half. The correct restatement is: **cluster aggregation is still a
production candidate, at a headline ratio of roughly 2–3× rather than 5.5–10.2×, measured
against a densified baseline that should be built either way.**

---

## 4. The work, in order

Each step is independently shippable and each ends in a measurement. The order differs from
the proposal's in three places, and each difference is a consequence of §1 or §3.

### Step 0 — correct what is now known to be false (half a day)

Not the proposal's general reframing; the two specific statements this document contradicts.

- `benchmarks/numba/timing.py::peak_allocation` docstring, and the parenthetical in
  `findings.md` §3.2 that repeats it (§1.3).
- `findings.md` §1's `parallel (4 cores)` column, split into total / parallel / efficiency
  (§1.4), and its `decision` column, rewritten to the proposal's vocabulary — *strong production
  candidate*, *promising prototype*, *retain numpy*, *reprofile after prerequisite fix*, *defer*
  — with `multiplier_bootstrap` moved to **fix in numpy first, numba unresolved** and
  `cluster_sums` restated against §3.3's baseline.
- The bootstrap paragraphs in `docs/roadmap.md` and the two-line summary in `README.md`, which
  both attribute the cost to the draw.

Acceptance: no benchmark-only kernel is described as adopted production architecture; every
memory claim states its instrument; §1.1's split appears wherever "92–95% generation" currently does.

### Step 1 — `thread_limit` (§3.1)

Largest measured effect in the package, no dependency, no statistical surface. Cache the
controller; invalidate where `library.py` imports a backend; test nesting, exception exits,
`set_thread_limit(None)`, fork and spawn workers, a backend imported after the first entry, and
a process with no supported pool loaded. Benchmark: bare entry, 1,000 entries, and a DR-TMLE
`retarget` before and after.

Acceptance: per-entry cost within a factor of two of the cached number above; limits still
effective under `threadpool_info()`; DR-TMLE and LTMLE reprofiled and the new profiles recorded.

### Step 2 — the multiplier bootstrap, in numpy (§1.1, §1.2)

Blocked expansion, in place, with the block size a tunable. Ship the float64 arm on its own —
bit-identical output, ~1.9× — and treat float32 as a **separate, argued** change with its own
tolerance analysis, since it moves the arithmetic of a reported quantile.

Acceptance: identical critical value under a fixed seed for the float64 arm, pinned by a test;
allocation per call down by the block ratio, measured; a test that the block size does not move
the seeded output; the production benchmark re-run through `multiplier_critical_value` itself.

### Step 3 — cluster label densification, in numpy (§3.3)

Densify at the container, carry `(codes, n_clusters)`, and make `cluster_sums` accept
already-dense codes. The output contract the proposal's §4.1 lists is the specification: label
ordering, the mapping from original labels to output rows, non-contiguous and string-derived
labels, dtype. Note that a densify-once changes **row order** if it is done in first-appearance
order — `np.unique` sorts — so it must preserve the sorted order, or every consumer must be
audited. Preserve it.

Acceptance: byte-identical output on every existing clustered test; the densify happens once per
fit rather than once per call, pinned by a call-count test; the bootstrap path measured.

### Step 4 — phase timing for LTMLE and survival (proposal §11)

Before the mask fix, not after it, because the mask term is invisible at the `T` the fixtures use
and the fix has to be justified at a realistic one. A disabled-by-default collector, wall clock,
per-node and per-regimen counts.

Acceptance: no phase share in any report is inferred from a cProfile filename match; a run at
`T ∈ {2, 5, 20}` shows the mask term's share directly.

### Step 5 — the longitudinal prefix masks (§3.2)

Prefix state carried down the nodes. The invariants are the ones `CLAUDE.md` already names —
`at_risk(t+1) == following(t)` on an end-of-study fit and its survival generalisation, the
event node one earlier than the censoring node, the mechanism's event-aware fit mask — and the
oracle laws are what enforce them. Compare the masks themselves, not only the estimates: an
end-of-study fit whose event can only happen at the last node must stay bit-for-bit identical.

Acceptance: mask-construction cost approximately linear in `T`, measured through Step 4's
instrument; every oracle law unchanged; a `T = 20` production fit improved by the share Step 4
reports.

### Step 6 — the kernel seam, and only then a compiled backend

`src/cleverly/_kernels/` as the proposal specifies it, with `backend="numpy"` the default,
numba optional and never imported at package import, and the benchmark calling the production
backend. Build it when there is a kernel that has cleared Steps 2 and 3 — which is a question
the measurement answers, not this document. On the evidence so far, cluster aggregation reaches
it and the bootstrap may not.

Acceptance: `pip install cleverly` still has no numba; `backend="numba"` and `backend="numpy"`
agree to a stated tolerance and do not move with the thread count; the benchmark's numpy arm
*is* the production numpy path.

### Step 7 — reprofile and decide the rest

DR-TMLE after Step 1, LTMLE and survival after Steps 4–5, and the deferred list re-read against
the new profiles. The proposal's 5%-of-corrected-runtime rule is the right bar.

---

## 5. Deliverables

Steps 0–5 produce no new package structure, which is the point of the reordering:

```
benchmarks/results/production_plan.md        this document
benchmarks/results/revised_findings.md       or a rewritten findings.md; one or the other, not both
benchmarks/results/thread_limit_profile.md   Step 1
benchmarks/results/bootstrap_numpy.md        Step 2 (the proposal's bootstrap_integration.md, renamed
                                             for what it will actually contain)
benchmarks/results/cluster_integration.md    Step 3
benchmarks/results/longitudinal_phases.md    Steps 4-5
tests/unit/test_thread_limit.py              Step 1
tests/unit/test_multiplier_blocking.py       Step 2
tests/unit/test_cluster_codes.py             Step 3
tests/unit/test_longitudinal_masks.py        Step 5
src/cleverly/_kernels/                       Step 6, conditional on Steps 2-3
```

`pyproject.toml`, `noxfile.py` and the CI workflows change only at Step 6, and `numba` moves out
of `[bench]` only if a kernel clears it.

---

## 6. What this plan does not answer

- **Whether a compiled bootstrap kernel beats §1.1's numpy baseline.** The one written for this
  document loses badly (§1.2), which is evidence about that kernel and not about the question.
- **The float32 question.** Whether a `float32` expansion is acceptable inside a Monte Carlo
  quantile is a statistical judgement with a 7.2×-against-1.9× price tag on it, and it is not
  settled by the 1e-6 agreement measured at one configuration.
- **Anything about a machine with more than four cores**, unchanged from `findings.md` §9.
- **The estimand-count axis at `m = 1`.** `multiplier_critical_value` returns `norm.ppf`
  immediately for a single estimand, so the proposal's `estimands = 1` benchmark point does not
  reach any kernel. Keep it as an API test.
