# Candidate inventory: what the post-nuisance profile actually says

The first deliverable of the numba investigation, and the one that decides where the rest
of the effort goes. Everything here is measured, on the box named at the bottom; nothing is
inferred from the shape of the code.

**The scope is the half of a fit this package owns.** Cached nuisance predictions →
targeting → estimands → influence curves → standard errors and inference. Nuisance
estimation is excluded: those are scikit-learn and LightGBM, they already run compiled, and
the question is not whether numba can beat them.

**Every share below is quoted against `library="glm"` unless it says otherwise, and that
is the *most favourable* denominator available.** `glm` is the cheapest preset the package
offers. A `default` fit costs roughly 37× more per row (`docs/roadmap.md`), so a line
reading 30% of a `glm` fit is under 1% of a fit anybody runs. Both columns matter and
neither is the answer on its own.

---

## 1. The ranked table

Runtime share is of the **post-nuisance region**, not of the fit, except where the column
says otherwise. "Repetition" is what a real workload multiplies the kernel by.

| estimator | function | runtime share | allocations | repetition | numba plausibility | parallel axis | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| inference | `multiplier_critical_value`, `kind="rademacher"` | **≈100% of a bands computation**; 14% of a whole `glm` LTMLE fit | a `(256, n)` float64 array per chunk — 200 MB at n=1e6 | B = 1,000–10,000 replicates | **high** — generation-bound, and the array is consumed once | replicates | **investigate** |
| inference | `cluster_sums` | 0.08–0.58 s at n=1e6 depending on estimand count; called once per estimate *and once per bootstrap draw* | `np.unique`'s sort, plus m separate outputs | per replicate | **high** — indexed accumulation, m passes over one index vector, a sort that buys nothing | clusters | **investigate** |
| tmle | `solve_one_step` | **31% of a `glm` fit** at n=20,000; the roadmap's asymptotic table puts it at **82%** | ~20 full-length passes and a dozen temporaries **per step**, up to 20,000 steps | every step of the walk | **high** — the interpreter is genuinely in the inner loop | rows (steps are sequential) | **investigate** |
| ltmle | backward recursion masks + fluctuation | 60% of a cached-nuisance recursion at T=20 | masks are `O(T² n)` as written; `O(T³ n)` under a per-horizon survival pass | one pass per regimen × cause × horizon | **high** — Python loop over nodes, boolean masks rebuilt | regimens | **investigate** |
| survival | per-horizon incidence recursion | the dominant package-owned cost of a survival fit | `T(T+1)/2` node passes per regimen per cause | horizons × causes × regimens | **high** — same recursion, run `T`× more often | horizons | **investigate** |
| cvtmle | `TMLE._solve_by_fold` | 6–16% of a `glm` fit; **grows with fold count** (72 ms at 2 folds → 156 ms at 20, n=20,000) | one `restrict`/`stitch` fancy-index per fold | folds | **medium** — the per-fold body is the Newton, which is BLAS-bound | folds (task-parallel) | **investigate the axis, not the arithmetic** |
| ctmle | candidate scoring | **11 ms of a 190 s fit** | — | candidates × folds | **low in situ, high in principle** | candidates (only when pre-enumerated) | **measure, expect a negative** |
| drtmle | reduction alternation arithmetic | ~40% of a `retarget`'s *non-external* time | two covariates + two fluctuations per round | rounds × arms | **low** | arms (2-wide) | **measure, expect a negative** |
| tmle | fused multi-estimand influence curves | **2–4 ms of a 28.6 ms `retarget`** (~10%) | one full-length array per estimand | once per fit | **low** — the ceiling is a tenth of the step before anything is fused | rows | **measure; the plan's headline hypothesis deserves a number** |
| tmle | targeting Newton (`_newton_logistic`) | 17% of a `retarget`; 0.31% of a `glm` fit | small | 1–3 iterations | **negative control** — `x @ eps`, `x.T @ (...)`, one vectorised `expit`: already BLAS and SIMD | none | **retain numpy** |
| msm | `_projection_state` Gram einsum | ~1% of a fit | — | per Newton step and line-search trial | **negative control** — one `dgemm` after `optimize=True` | none | **retain numpy** |
| *all* | **`cleverly.learners.thread_limit`** | **57% of a DR-TMLE `retarget`; 40% of an LTMLE fit** | — | **once per learner fit** | **none — there is nothing to compile** | none | **fix in numpy-land; see §4** |

---

## 2. Where each number came from

### 2.1 Point-treatment TMLE

`TMLE.retarget` is the package's own name for the primary denominator, so it is measured
directly. Binary outcome, n = 20,000, `library="glm"`, `n_folds=2`:

| requested estimands | `retarget` | share of the 0.52 s fit |
| --- | --- | --- |
| 1 (`ate`) | 11.5 ms | 2.2% |
| 3 (`ey1`, `ey0`, `ate`) | 12.3 ms | 2.4% |
| 5 (+ `att`, `atc`) | 26.1 ms | 5.0% |
| 7 (+ `rr`, `or`) | 28.6 ms | 5.5% |

**The jump is between three and five, and it is not the influence curves.** `att` and
`atc` are separate *target groups*, and each one pays its own fluctuation solve; `rr` and
`or` are delta-method transforms of estimates that already exist and are nearly free. So
the estimand count is a proxy for the target-group count, and the plan's hypothesis that
fusing the curves is where the money is does not survive its own profile:
`counterfactual_means` and `_conditional_effects` together are 2–4 ms of the 28.6.

Inside a `retarget` at 7 estimands (cProfile, `tottime`, 10 calls):

| function | tottime | share |
| --- | --- | --- |
| `_newton_logistic` | 0.047 s | 16% |
| `numpy.ufunc.reduce` | 0.026 | 9% |
| `bounds.expit` | 0.023 | 8% |
| `_score.score_scale` | 0.020 | 7% |
| `_score.score_columns` | 0.019 | 6% |
| `_score.quasi_loglik` | 0.019 | 6% |
| `numpy` clip | 0.015 | 5% |
| `solve_fluctuation` | 0.012 | 4% |
| `apply_logistic` + its dictcomp | 0.017 | 6% |
| `counterfactual_means`, `_conditional_effects` | 0.006 | 2% |

### 2.2 The one-step walk

The same fit with `targeting="one_step"`: `retarget` is **0.155 s, 31% of the fit**, and
the profile is a different shape entirely —

| function | tottime (3 calls) | share |
| --- | --- | --- |
| `apply_logistic`'s per-arm dictcomp | 0.092 s | 18% |
| `score_columns` | 0.061 | 12% |
| `bounds.logit` | 0.059 | 12% |
| `bounds.expit` | 0.057 | 11% |
| `apply_logistic` itself | 0.048 | 9% |
| `numpy.ufunc.reduce` | 0.047 | 9% |
| `numpy` clip | 0.043 | 8% |

Every one of those is a full-length array pass, and the walk does about twenty of them per
step. This is the one place in the package where the interpreter is in the inner loop.

### 2.3 CV-TMLE

Fold-specific targeting, n = 20,000, `library="glm"`:

| folds | `retarget` | share of fit |
| --- | --- | --- |
| 2 | 72.5 ms | 16.2% |
| 5 | 85.7 ms | 10.4% |
| 10 | 115.8 ms | 8.5% |
| 20 | 156.1 ms | 6.2% |

The share falls because the *fit* gets more expensive faster than the targeting does; the
absolute cost more than doubles. `_solve_by_fold` is a plain serial `for` over
`nuisance.folds`, and the folds are independent by construction — a fold's `epsilon` is fit
only against rows whose predictions came from models that never saw them. That is a
task-parallel axis sitting unused. What is *inside* each fold, though, is the Newton solve,
which the control below says is not worth compiling.

### 2.4 CTMLE — the premise does not hold

One `CTMLE.fit` at n = 20,000 with a `glm` outcome learner and the default treatment
library takes **190 seconds**. The largest lines:

| line | tottime |
| --- | --- |
| `lightgbm.basic.update` | 23.2 s |
| `sklearn.svm._base._fit_liblinear` | 15.8 s |
| `numpy._unique_hash` | 8.1 s |
| `lightgbm.__inner_predict_np2d` | 5.7 s |
| `inspect._shadowed_dict` (via sklearn validation) | 5.4 s |
| `sklearn.utils.validation.check_array` | 2.7 s (35.4 s cumulative) |
| `threadpoolctl` | ~3.5 s |

The post-selection `retarget` — the whole of what this package owns once the candidates
exist — is **11 ms**. The candidate search is candidate-*fitting*-bound. The plan's
reasoning ("it evaluates many candidates repeatedly") is right about the repetition and
wrong about what is being repeated.

### 2.5 DR-TMLE — the largest single finding, and it is not about numba

`DRTMLE.retarget` at n = 20,000, `glm`, three estimands: **16.1 s.** Against a fit that
measured 7.3 s in the first profile and 10.5 s in the reproducible pipeline run — the fit
time is the noisy half of that ratio on a shared box; the `retarget` is 16.1–16.2 s in both.
Either way it is a `retarget` that costs 1.5–2.2× the fit it is meant to be a cheap re-run
of. Reproduce with `python -m benchmarks.numba.cli --pipelines drtmle`.

The alternation legitimately refits its reduced regressions — `g_{r,2}` is a functional of
the mechanism being tilted, so `ReductionSpec.refit` is correct. What is not legitimate is
where the time goes. Profiling three such calls (71.9 s total):

| line | tottime | calls |
| --- | --- | --- |
| `threadpoolctl._check_prefix` | 9.45 s | 15,602,160 |
| `threadpoolctl._make_controller_from_path` | 8.42 s (34.0 s cumulative) | 3,120,432 |
| `str.startswith` | 5.75 s | 41,017,608 |
| `threadpoolctl.match_library_callback` | 3.30 s (37.9 s cumulative) | 3,132,864 |
| `threadpoolctl._find_libraries_with_dl_iterate_phdr` | 2.80 s (**40.8 s cumulative**) | 12,432 |

**57% of a DR-TMLE `retarget` is `threadpoolctl` walking the process's loaded shared
objects.** `cleverly.learners.thread_limit` constructs a fresh `ThreadpoolController` on
every entry, and it is entered once per `fit_learner` call. Measured directly on this box:
**1.44 ms per entry** (a bare `threadpool_limits(limits=1)` is 0.22 ms — the rest is the
controller construction). At thousands of small fits per `retarget`, the thread-limiting
costs several times the fitting.

Same signature in LTMLE: a 2.07 s fit at n = 20,000 spends **0.84 s cumulative** inside
`threadpoolctl` (248 `dl_iterate_phdr` walks) against 0.043 s in `loss_gradient`.

### 2.6 LTMLE

The 2.07 s `glm` fit above, by line:

| line | tottime | share |
| --- | --- | --- |
| `multiplier._multipliers` | 0.289 s | 14% |
| `threadpoolctl` (cumulative) | 0.838 s | 40% |
| `sklearn._loss.loss_gradient` | 0.043 s | 2% |

The largest *arithmetic* line in a longitudinal fit is the Rademacher multiplier draw
(`LTMLE` defaults to `n_multiplier=2000`), and the largest line of any kind is the thread
limiter. The recursion itself does not appear. That is consistent with the roadmap's
prediction that LTMLE is scikit-learn-bound — but the binding constituent turns out to be
the machinery around the learner rather than the learner.

### 2.7 The inference kernels, measured on their own

`cluster_sums` (ms, this box):

| n | m=1 | m=5 | m=20 |
| --- | --- | --- | --- |
| 20,000 | 0.6–0.8 | 1.0–1.2 | 2.7–4.4 |
| 100,000 | 2.8–4.5 | 4.2–12.6 | 15.8–59.2 |
| 1,000,000 | 78–111 | 115–153 | 468–576 |

Very nearly linear in the estimand count, which is the signature of "one `bincount` pass
per column over the same index vector". The cluster count moves it much less than the
estimand count does.

`multiplier_critical_value` (ms):

| n | B | rademacher | normal |
| --- | --- | --- | --- |
| 20,000 | 2,000 | 423 | 2.1 |
| 100,000 | 2,000 | 1,721 | 5.8 |
| 100,000 | 10,000 | 8,257 | 7.1 |

`_multipliers((256, 100000), "rademacher")` alone is **207 ms**, and 40 such chunks make
the 8.26 s. The draw is essentially the whole cost, exactly as the roadmap recorded, and
the `(256, n)` float64 array it produces is consumed once.

---

## 3. What qualifies, and what each candidate is qualified *by*

The plan's criteria, applied:

| candidate | ≥5% of the region | called many times | large temporaries | Python loop | indexed accumulation | independent axis |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| multiplier bootstrap | ✓ | ✓ | ✓ (200 MB) | — | — | ✓ replicates |
| `cluster_sums` | ✓ | ✓ | — | — | ✓ | ✓ clusters |
| one-step walk | ✓ | ✓ | ✓ (dozens per step) | ✓ | — | ✓ rows |
| LTMLE recursion | ✓ | ✓ | ✓ | ✓ | — | ✓ regimens |
| survival incidence | ✓ | ✓ | ✓ | ✓ | — | ✓ horizons |
| CV-TMLE fold loop | ✓ | ✓ | — | ✓ | — | ✓ folds |
| CTMLE scoring | — | ✓ | — | ✓ | — | ✓ candidates |
| DR-TMLE alternation | ✓ | ✓ | — | ✓ | — | ✓ arms (2-wide) |
| fused influence curves | — | — | ✓ | — | — | ✓ rows |
| Newton solve | ✓ | ✓ | — | — | — | — |
| MSM Gram | — | ✓ | — | — | — | — |

The last two are the negative controls. They stay in the suite: a benchmark that drops the
kernels it expected to reject cannot be used to check that its accepted ones are not an
artefact of the harness.

---

## 4. The finding that is not a numba finding

`cleverly.learners.thread_limit` is entered once per learner fit and costs **1.44 ms** per
entry on this box, because `threadpoolctl.threadpool_limits` builds a `ThreadpoolController`
that walks every shared object the process has loaded. That is 57% of a DR-TMLE `retarget`
and 40% of an LTMLE fit.

Nothing in this investigation addresses it — there is no arithmetic to compile and no axis
to parallelise. It is recorded here because a benchmark suite that reported only the
computations it *could* compile would have left the largest package-owned cost in two of
the seven flavours unmeasured, and because the fix is cheap and belongs to the same
"improve the numpy instead" category as the other algorithmic findings in this report:
construct the controller once and reuse it, rather than per fit.

It is deliberately **not fixed as part of this work**. Changing when and how thread limits
are applied is a change to the library's runtime behaviour with its own correctness
surface — `set_thread_limit(None)`, nested fits, joblib workers — and it belongs in its own
change with its own tests, not folded into a benchmark. `benchmarks/numba/kernels/drtmle.py`
carries `thread_limit_overhead()` so the number can be reproduced in one call.

---

## 5. What the profile changed about the plan

Three of the plan's expectations did not survive contact with a measurement, and saying so
is the point of doing this first:

1. **Fused multi-estimand influence curves are not the headline.** They are about a tenth
   of a cached-nuisance `retarget`. The estimand count is a proxy for the *target-group*
   count, and each group's cost is its fluctuation. Benchmarked anyway; the result is
   in `summary.md`.
2. **CTMLE is not a strong candidate.** Its candidate path is candidate-fitting-bound by
   three orders of magnitude. The scoring kernel is still worth a number — it is what would
   matter if candidates were cheap — but the framing "numba could speed up CTMLE" is wrong
   about where CTMLE's time is.
3. **The largest package-owned costs are not arithmetic at all.** In DR-TMLE and LTMLE they
   are the thread limiter; in the one-step walk and the LTMLE recursion they are algorithmic
   redundancies (counterfactual arms recomputed per trial step, masks rebuilt per node) that
   a compiler helps with only incidentally. Every one of those has a numpy-side arm in the
   benchmark for exactly this reason.

---

## 6. Provenance

Measured on the four-core container this repository's cloud sessions run in:

- **CPU**: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 physical / 4 logical cores
- **BLAS**: OpenBLAS 0.3.31 (pthreads), SkylakeX kernels
- **Python** 3.11, **numpy** 2.4.6, **numba** 0.66.0, **scikit-learn** 1.9.0, **LightGBM** 4.7.0

A four-core shared container is a poor place to measure parallel scaling past two threads
and a fine place to measure a serial speed-up or a profile share. Read the scaling tables
in `summary.md` with that in mind, and re-run `benchmarks/configs/full.yaml` on a machine
with cores to spare before quoting an efficiency figure.

Reproduce with:

```bash
pip install -e '.[bench]'
python -m benchmarks.numba.cli --config benchmarks/configs/sandbox.json
python -m benchmarks.numba.cli --pipelines --pipeline-libraries glm default
python -m benchmarks.numba.cli --cold-compile
```
