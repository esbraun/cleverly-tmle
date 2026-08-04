# Benchmarks: where the time goes, and what compiling it would buy

> ## The decision, in one paragraph
>
> **`numba` is a benchmark-only dependency and nothing under `src/` imports it.** Every one
> of the three "adopt numba" recommendations this investigation produced turned out to be a
> *numpy* result once the baseline was written properly: an expansion that did not need
> doing, a sort that had already been done elsewhere, and a mask rebuild that was
> quadratic. The largest single win in the whole exercise — 49% of a `DRTMLE` `retarget` —
> was a context manager. What would reopen the question is a kernel that beats a
> **competent numpy baseline**, measured on a machine with more than four cores; nothing
> here has been measured above four, and no CI job runs above two.
>
> The roadmap's [standing decisions](../roadmap.md#standing-decisions) is the register.
> This directory is its evidence.

## The documents, in the order they were written

| document | what it is |
| --- | --- |
| [`candidate_inventory.md`](candidate_inventory.md) | **the profile**, and what the rest was sized against. Read it first: three of the things it is natural to expect turn out to be false, and the largest package-owned cost in two flavours is not arithmetic at all |
| [`findings.md`](findings.md) | **the measurement**: twelve kernels, serial and parallel, with the controls, the memory column, the compile amortisation, and the share of a fit each speed-up has to be multiplied by |
| [`production_plan.md`](production_plan.md) | **the adjudication** — what to build, and where the plan it revises was wrong. Its §1 is four claims contradicted by measurements rather than by arguments |

And the four results of executing that plan, each a step of its §4:

| document | what it found |
| --- | --- |
| [`thread_limit_profile.md`](thread_limit_profile.md) | building `threadpoolctl`'s controller once per process rather than once per learner fit: **59× per entry**, 49% of a DR-TMLE `retarget` |
| [`bootstrap_numpy.md`](bootstrap_numpy.md) | the multiplier bootstrap rewritten in numpy: **3.4–3.9×**, and 1,881 MB down to a 32 MB budget at `n = 10⁶`, with the seeded stream bit-identical |
| [`cluster_integration.md`](cluster_integration.md) | `cluster_sums` against the codes the container actually produces, where the compiled kernel's advantage largely disappears |
| [`longitudinal_masks.md`](longitudinal_masks.md) | the LTMLE mask rebuild, `O(T²n)` → `O(Tn)` — real, and 0.06% of a fit |

## Reading a number out of any of them

Three things a reader has to know, and each has cost someone a wrong conclusion here.

**A ratio measured against the shipped shape is not a ratio against numpy.** This is one
mistake made three times, and it is what `production_plan.md` §1 is about. Before quoting a
speed-up, check what a competent numpy version of the same function would cost.

**A kernel-level ratio is not a fit-level one.** Everything here is *post-nuisance*: the
learner fits are outside every timed region by construction, which is what makes `n = 10⁶`
a couple of seconds. `findings.md` §5 is the denominator, and a 10× on 1.5% of a fit is not
a dependency.

**The numbers are from the box they were taken on, and from the harness of the day.**
Provenance blocks name the machine. They now also have to name the *method*: everything
recorded here was measured under randomised block order, and the harness has since moved to
a genuine rotation, so a rerun is a different instrument rather than a replication.

## Running one

```bash
pip install -e '.[bench]'
nox -s bench-numba              # the kernels
nox -s bench-numba-pipelines    # the denominator: post-nuisance share, per flavour
```

Runs write to `benchmarks/results/`, which is git-ignored output. `benchmarks/bench_tmle.py`
is the other instrument and answers a wider question — where a *whole fit's* time goes,
learners included.
