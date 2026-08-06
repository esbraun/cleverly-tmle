# The component-level trace, and what a first run of it says

[The roadmap](../roadmap.md)'s [piece F](../roadmap.md#f-localize-the-shortfall-before-changing-anything)
is a recovery plan whose premise is that the `DRTMLE` shortfall is **measured and not
localized**, and whose order puts **F2 second**: it "produces the common state-level
instrument F3 and F4 both read". This document is that instrument's record —
`benchmarks/drtmle_trace.py`, its frozen fixture, and the four things running it has already
made visible.

**It closes nothing, and that is what it is for.** Nothing under `src/` moves here; only
[F7](../roadmap.md#f-localize-the-shortfall-before-changing-anything) may. A divergence this
makes visible is a *question*, adjudicated against Theorem 1,
[the concordance](theorem-concordance.md), the exact-law identities and the remainder
decomposition — never settled by which side another implementation is on
([stop-ship 17](../roadmap.md#stop-ship)).

## What it is

| piece | what it holds |
| --- | --- |
| `benchmarks/fixtures/drtmle_trace_v1.csv` | 200 rows: `w1`, `w2`, `a`, `y`, `fold`, `weight`, and the **initial** `qn1`, `qn0`, `gn` |
| `benchmarks/fixtures/drtmle_trace_v1.json` | the SHA-256 of those bytes, the closed-form coefficients, the bounds, the fold count, the schema version |
| `benchmarks/drtmle_trace.py` | the injected learners, the tracing estimator, the step record, the identities, and the CLI |
| `tests/unit/test_drtmle_trace.py` | the guards that make it an instrument rather than a printout |

```bash
python -m benchmarks.drtmle_trace --both            # both update orders, compared
python -m benchmarks.drtmle_trace --order paper --out benchmarks/results/trace
python -m benchmarks.drtmle_trace --write-fixture   # regenerate; every trace already taken
                                                    # is against the old bytes
```

## The four fixture choices, and why each is load-bearing

**The outcome is binary, so the outcome scaler is the identity.**
`benchmarks/drtmle_injection.py` records the reason it matters: recovering the affine map for
a continuous outcome carries an `O(n^{-1/2})` error, which is exactly the thing "identical
initial `Q̄`" between two implementations would break. A binary `Y` removes the map rather
than estimating it, so `qn1`/`qn0` are the arrays both sides start from with nothing in
between.

**Both initial nuisances are misspecified, deliberately.** At correct nuisances `Q_r` and
`g_{r,2}` vanish row by row, both corrections are zero arrays, and the reported curve equals
`D*` array for array — so a trace taken there passes against a flipped sign, a swapped update
order and a stale reduction alike. That is `CLAUDE.md`'s rule about where an exact-law
instrument goes blind, in the place it bites hardest, and it is why
`degeneracy()` is a reported quantity and a test rather than a remark. On this fixture:

| quantity | `cleverly` | `paper` |
| --- | --- | --- |
| `max\|Q_r\|` | 0.2806 | 0.6834 |
| `max\|g_{r,2}\|` | 0.5594 | 0.4242 |
| `max\|D*_g\|` | 0.4047 | 1.688 |
| `max\|D*_Q\|` | 0.3981 | 0.2769 |

against a `mean|Y|` of 0.575.

**The truncation is slack on every row.** Clipping is not irrelevant — it is
[item 20](../roadmap.md#what-is-still-open) and the whole of piece B1b — but the two
implementations' truncation *conventions* differ, and a first-divergence hunt confounded by a
known convention difference locates the convention rather than the defect. The trace reports
`clipped = 0` so a reader sees it rather than assumes it. **A fixture that turns clipping on
is a second fixture**, not an edit to this one.

**`n = 200` is the fast tier's budget, and the neighbouring sizes are not neighbouring
experiments.** Measured on four cores: `n = 200` is 3.2 s under `cleverly` and 6.5 s under
`paper`; `n = 300` is 4.2 s and 10.6 s. At `n = 250` the same recipe takes **43 rounds** and
47 s under `paper` and exits on a *stall* rather than the tolerance under `cleverly`. That is
equation (10)'s conditioning — this package's own docstrings describe it as near-singular on
exactly the fits anybody wants — and it is the reason a trace is taken on **one frozen draw**
rather than on a size sweep.

## The step vocabulary F3 aligns against

| equation | what it is |
| --- | --- |
| `8` | the outcome fluctuation along `1_a/g*` |
| `9` | the mechanism tilt along `Q_r/g*` |
| `10` | the outcome fluctuation along `g_{r,2}/g_{r,1}` |
| `refit` | a re-estimation of the reduced regressions at the current pair |
| `joint` | the closing pass's four-column solve of (8) and (10) together — the R package has no analogue |

with `phase ∈ {prime, round, close}`. `prime` is the equation-(8) solve before the loop;
`close` is `_close_at_frozen_reductions`, whose boundary **no field on a returned fit
distinguishes** and which F4's pre-close/post-close column needs.

Every step carries the whole state either side of it — `Q*` at both arms and at the observed
one, `g*`, `Q_r`, `g_{r,1}`, `g_{r,2}` — plus `epsilon`, the recorded score, the Newton count,
the Hessian condition and the failure flag.

## What the first run measured

### 1. Every recorded score recomputes from the recorded state

62 recomputations under `cleverly` and 167 under `paper`, longhand in the harness rather than
through `score_columns`, worst residual **1.11e-16** against `IDENTITY_TOLERANCE = 1e-12`.
That is F2's acceptance criterion and it is met.

**Those three numbers are one run's observation and not a committed record**, and the difference
matters to anyone quoting them. No trace artefact is in the tree — `benchmarks/results/` holds
a `.gitkeep` — and the test asserts the *worst residual* against `IDENTITY_TOLERANCE` and that
all five quantity families appear, never the counts. Rerun `--both` and the counts are what the
alternation's round count makes them. What is pinned is the tolerance and the coverage; the
tallies are a reading.

The recomputation reads the **covariate off the state the step started from and the fitted
value off the state it left**, and that asymmetry is the content of the check. A solver zeroes
its score at the covariate it was handed against the fit it produced; recomputing both halves
at one state gives a different number and would read as a defect. It is the distinction item
20 turned on, where equation (9) was solved at the raw tilt while the curve subtracted the
truncated one.

### 2. The two update orders exit on `tolerance` and do not land in the same place

Both routes report `exit_reason = "tolerance"` on this draw, and:

| | `cleverly` | `paper` | difference |
| --- | --- | --- | --- |
| `psi[ate]` | 0.214075 | 0.207703 | `+0.00637`, **0.098 standard errors** |
| `se[ate]` | 0.06522 | 0.065078 | `+1.4e-04` |
| `psi[ey0]` | 0.476179 | 0.480796 | `−0.00462`, −0.097 se |
| rounds | 5 | 11 | |
| closing solves | 4 | 21 | |

and the **curves disagree far more than the standard errors do**. For `ate`, `sd(D) = 0.922`
while `max|ΔD| = 1.341` — one and a half standard deviations of the curve at a single row —
with an rms difference of `0.183`, a fifth of `sd(D)`. `ey0` is the same shape (`max|ΔD| =
1.340` against `sd(D) = 0.671`).

**This is one draw and it is not a verdict.** [Item 22](../roadmap.md#what-is-still-open) asks
whether the two routes reach the same fixed point on real data; a single fixture supplies a
data point and not an answer, and `compare()` reports the differences rather than asserting
them away. What it does establish is that the question is not idle at this scale: the routes
differ by a tenth of a standard error in the point estimate and by more than a standard
deviation of the curve pointwise, while both report having reached the tolerance.

### 3. The closing pass's mechanism stage can be stationary and still spend its whole budget

**That the cap binds is not news.** The roadmap's B2b dispatch measured it at **96 of 96**
fits and predicted it in advance; it is the one place a bounded truncation convention cost
anything. What is new is *why*, which no field on a fitted result distinguishes.

Under `paper`, `_close_at_frozen_reductions` runs its equation-(9) stage to the `max_steps =
20` cap and reports `closing_capped = True`. **14 of those 20 steps return a coefficient of
exactly zero and leave the score unchanged.** The stage is at a fixed point after the sixth
step; what the cap means here is not "still converging" but "stationary at a point the exit
test cannot recognise" — and those are different facts about one flag.

That is expected in *kind* — the function's own docstring says equation (9)'s covariate reads
the very mechanism it tilts, so a solve leaves a residual at the post-tilt covariate that
iterating shrinks without removing — but the docstring says *shrinks*, and on this fixture it
does not shrink at all after the sixth step. Under `cleverly` the same stage takes 3 steps and
converges.

**Recorded, not fixed.** An early exit when a step returns a zero coefficient would cost
nothing numerically and would take this stage from 20 solves to 6 on a fit like this one, but
it is a change under `src/` and therefore F7's, on a localization the ablations support — not
F2's.

### 4. The reduction-refit vintage is now readable off a run

The fourth row of the roadmap's [R3 table](../roadmap.md#f-localize-the-shortfall-before-changing-anything)
— *"refits all reductions again"* against *"refits `Qr` after both outcome updates"* — is one
of the five places two correct-looking implementations of this algorithm can differ, and it is
**invisible in every field a fitted result carries**. `vintages()` reads it off the trace:
each `refit` step is compared against what the *next* equation's covariate was actually built
from.

| order | reading |
| --- | --- |
| `cleverly` | every refit: `qr`, `gr1`, `gr2` all adopted — one vintage per round |
| `paper` | alternating `qr=False, gr1=True, gr2=True` and `qr=True, gr1=False, gr2=False` — **two vintages per round**, exactly the paper's steps 3 and 5 |

## Three things the harness got wrong first, kept here because they are the class of error F3 must not make

**Reading the reductions off the refit closure rather than off the covariate builders.** The
refit returns all three regressions; the paper order then adopts `gr1`/`gr2` from one call and
`qr` from another. A recorder that took the closure's output as the state scored every
equation against a set the estimator never used — and it *passed* under `cleverly`, where the
two coincide. Under `paper` the closing pass's equation-(10) identity missed by **1.3e-08**
against a bar of `1e-12`. The fix is that the recorder's reduced state is written where each
covariate is built, so it is the set the equation actually read.

**pandas' default CSV float parser is fast rather than exact.** Written with
`float_format="%.17g"` and read back with the default parser, `w1` came back short by one unit
in the last place on **65 of 200 rows**, at `2.2e-16`. Both halves are fixed — the shortest
round-trip repr on write, `float_precision="round_trip"` on read — and the reason it matters
is that `2.2e-16` in the inputs is precisely the size of difference a first-divergence hunt
would find between two implementations and mis-classify as a learner difference. The harness
would have manufactured the divergence it was built to locate.

**A trace was labelled by a module global rather than by the fixture it ran on**, and that one
arrived with `v2` rather than before it. `write_trace` built both filenames from
`FIXTURE_VERSION = "v1"` and `digest` never carried the fixture at all, so a trace taken on the
bound-active fixture was written to `drtmle_trace_v1_*` — and this module's own CLI had no
version flag, so `v2` was reachable from `drtmle_r_compare` and from nowhere here. `Trace` now
carries `fixture_version`, both filenames and the digest payload read it, and `--fixture-version`
exists. It is the same class as the other two: not a wrong number, a **right number filed under
the wrong experiment**, which is worse because nothing downstream can tell.

## What the tests guard

`tests/unit/test_drtmle_trace.py`, 30 tests, ~27 s — five `DRTMLE` fits and no more, each
answering a different question.

- **the fixture is frozen** — the digest matches the manifest, the draw regenerates from its
  seed, the `qn1`/`qn0`/`gn` columns are the closed forms F3 hands R, the committed `fold`
  column is the one a fit realises, and the truncation is slack on every row;
- **the instrument does not move what it measures** — a traced fit is bit-for-bit an untraced
  one, `psi`, `se`, every curve and every exit field, under **both** orders; and the patched
  module is left unpatched afterwards;
- **the trace is deterministic** — a second run digests identically, checked on SHA-256 rather
  than with a tolerance, because a tolerance here would hide the drift it exists to catch;
- **every recorded score recomputes**, all five quantities present so a mis-classified step
  fails rather than drops out of the check;
- **the identities can fail** — a perturbed recorded state is required to break them, which is
  what makes agreement evidence rather than a tautology;
- **the fixture is not degenerate** — every one of the four quantities above is required to be
  material against `mean|Y|`;
- **a written trace says which fixture it is** — the two filenames and the digest payload carry
  the traced fixture's version under both `v1` and `v2`, asserted on the recorded *field* rather
  than on the arrays, since two different draws would differ anyway and a hash comparison would
  pass with the provenance missing.
