# The thread limiter, before and after

The largest package-owned cost `benchmarks/numba/` found, and the one that is not a
compilation question. [`candidate_inventory.md`](candidate_inventory.md) §4 records it and
declines to fix it inside a benchmark; this is the fix and its measurement.

`cleverly.learners.thread_limit` wraps every nuisance fit and every prediction so that the
parallelism happens across folds and candidates rather than inside a learner. It used to
call `threadpoolctl.threadpool_limits`, which **constructs a fresh `ThreadpoolController`
on every call** — and constructing one walks every shared object the process has loaded
(`dl_iterate_phdr`, then a prefix match per library). That walk is a fixed cost per entry,
and the entries are counted in thousands.

The change is to build one controller for the process and reuse it.

> Measured on the four-core Intel Xeon @ 2.80 GHz container this repository's cloud
> sessions run in, `/proc/loadavg` under 0.6, Python 3.11, numpy 2.4.6, threadpoolctl
> 3.6.0, with OpenBLAS ×2 and OpenMP loaded. Medians of interleaved repetitions.

## Per entry

| | ms per entry |
| --- | ---: |
| `with thread_limit(1)` — building a controller each time | **0.759** |
| `with thread_limit(1)` — reusing one | **0.0129** |
| | **58.8×** |

| repeated entries | before | after | |
| ---: | ---: | ---: | ---: |
| 100 | 0.082 s | 0.001 s | 61× |
| 1,000 | 0.849 s | 0.013 s | 65× |

The 0.759 ms here is smaller than the 1.44 ms `candidate_inventory.md` reports, and the
difference is the point rather than a discrepancy: the walk's cost scales with the number
of loaded shared objects, and that measurement was taken in a process that had also
imported LightGBM. A real fit has imported more, so the per-entry saving in situ is larger
than the one above.

## Through the shipped API

The number that matters is not the context manager's, it is the estimator's.

| | before | after | |
| --- | ---: | ---: | ---: |
| `TMLE.fit`, `glm`, n = 20,000, 7 estimands | 0.338 s | 0.271 s | **1.25×** |
| `LTMLE.fit`, `glm`, n = 20,000, T = 2 | 0.871 s | 0.648 s | **1.34×** |
| `DRTMLE.retarget`, `glm`, n = 5,000, 3 estimands | 5.206 s | 2.674 s | **1.95×** |

**49% of a DR-TMLE `retarget` was the thread limiter**, and it is now gone. That is against
the 57% the cProfile in `candidate_inventory.md` §2.5 attributed to `threadpoolctl`
cumulatively; a profiler charges per-call overhead to code that makes many small calls, so
the profile overstated it, and 49% measured without a profiler attached is the number to
quote. The same correction applies to the 40% quoted for an LTMLE fit, where the
wall-clock saving is 25%.

`DRTMLE.retarget` still costs about half its own fit rather than being the cheap re-run it
is meant to be. What is left is the alternation's genuine refitting of the reduced
regressions, which is correct work — `g_{r,2}` is a functional of the mechanism being
tilted — and `findings.md` §2.7's conclusion is unchanged by this: the remaining arithmetic
is not where a compiled kernel would pay.

## What the cache trades away, and how that is handled

A controller records the pools that existed **when it was built**. One thing in this package
loads a pool later: `cleverly.learners.library.has_lightgbm` imports LightGBM lazily, inside
the function that builds the learner, which brings an OpenMP runtime with it. A controller
cached at the first fit can therefore predate the pool it is supposed to limit.

Nothing detects that automatically, and nothing should try — the detection *is* the walk the
cache exists to avoid. So the invalidation is explicit and lives at the one place that loads
a backend: `has_lightgbm` calls `refresh_thread_pools()` on the import, once, and caches its
own answer so it does not call it again. `refresh_thread_pools` is public for a caller who
`dlopen`s something itself.

Three other cases resolve without a hook, and `tests/unit/test_thread_limit.py` pins the
behaviour rather than the reasoning:

- **A forked child** inherits the parent's mappings, so the cached controller's handles stay
  valid. A `register_at_fork` hook drops the cache in the child anyway, which costs one
  deferred walk and covers the child that goes on to load something the parent never had.
- **A spawned child** re-imports the module and starts with no cache.
- **Concurrent entry** from a thread-backed joblib is guarded by a lock, so four threads
  arriving together build one controller rather than four.

The eleven tests cover the limit being applied and restored, restoration after an exception,
nesting (the inner exit restores the *enclosing* limit, not the original), `set_thread_limit(None)`,
an explicit override of the configured default, the refresh mechanism, the `has_lightgbm`
call site, concurrency, and a build with no threadpoolctl at all.
