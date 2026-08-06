# The bounded differential run against the R package, and where the two constructions part

[The roadmap](../roadmap.md)'s [piece F](../roadmap.md#f-localize-the-shortfall-before-changing-anything)
is a recovery plan whose premise is that the `DRTMLE` shortfall is **measured and not
localized**. Its order puts **F3** third, beside F4: *"localize construction differences
before any code changes"*. This document is F3's record — what the run is, what it refuses,
and what it found.

**A divergence here is a question, not a verdict.** It is adjudicated against Theorem 1,
[the concordance](theorem-concordance.md), the exact-law identities and the remainder
decomposition — never settled by which side R is on. Changing this package to match R is
[stop-ship 17](../roadmap.md#stop-ship), agreement is not a release criterion, and
[`CLAUDE.md`'s narrowing](../roadmap.md#a-differential-diagnostic-against-r-refused-then-authorized)
says exactly how much of the standing refusal this takes with it: the epistemic half, none of
it. What the authorization buys is a way to *localize*; what it does not buy is a way to
decide.

Nothing under `src/` moved. Only [F7](../roadmap.md#f-localize-the-shortfall-before-changing-anything)
may, and only on a localization F3 to F5 produced.

## What it is

| piece | what it holds |
| --- | --- |
| `benchmarks/r/drtmle_reference.R` | the R side: the fixture's digest checked, `drtmle`'s own loop run with its internals wrapped, the step stream and the whole state exported as raw float64 |
| `benchmarks/drtmle_r_compare.py` | the Python side: the reader, the seven gates, the classification, the report |
| `.github/workflows/drtmle-r-differential.yml` | where R lives — dispatch only, nowhere else in the repository |
| `tests/unit/test_drtmle_r_compare.py` | 18 tests, ~3 s, **no R installed and none invoked** |

```bash
Rscript benchmarks/r/drtmle_reference.R --out benchmarks/results/r-trace --qsteps 2
python -m benchmarks.drtmle_r_compare --export benchmarks/results/r-trace \
    --out benchmarks/results/r-differential.md
```

## Four choices that are load-bearing, and each is one F2 already had to make

**The package's own loop runs; nothing is re-implemented.** The R script replaces `drtmle`'s
internals with wrappers that call the originals and record either side of them, and restores
them in `on.exit` — the R idiom for exactly what `TracingDRTMLE` does in Python. A replay of
the loop written in the harness would be a second implementation, and a first-divergence hunt
whose instrument is a re-implementation finds the instrument. That the wrapping does not move
the fit is checked rather than argued: the script refits with the wrappers off and compares
`psi` and `cov` at `1e-12`, and **fails closed** if they moved.

**Arrays cross the language boundary as raw little-endian float64, not as text.** F2's own
record is why: written at 17 significant digits and read back with a fast parser, the
fixture's `w1` came back short by one unit in the last place on 65 of 200 rows, at `2.2e-16` —
precisely the size of difference this run would find and mis-classify as a learner difference.
A binary blob has no parser to be inexact. The index beside it is scalars only, where CSV is
safe.

**The folds are handed over, not redrawn.** Two independent random splits would make every
reduced regression differ at gate 1 and end the comparison there, on a difference nobody was
asking about. `drtmle`'s `make_validRows` accepts a *vector* of fold assignments as well as a
count, so R is given the fixture's committed `fold` column — reached by wrapping that function,
because the exported `drtmle()` cannot pass a vector through (see the last section but one).

**The reduced learner is a bare unpenalised GLM on both sides.** The frozen trace's
`reduced_*_learner="glm"` is a **two-candidate Super Learner** over `{mean, glm}`, and
`stats::glm` is one unpenalised fit; a convex combination against a single fit is a learner
difference this run already knows about. So both sides are given the bare GLM —
`LinearRegression` is `stats::glm(family = gaussian)` exactly and agrees with it at `1.1e-15`;
`LogisticRegression` is the binomial one once its solver is actually run to convergence, at
`3.7e-10` — matching `glm_Qr = "gn"` and `glm_gr = "Qn"` term for term. What the *shipped* reduced learner does instead is
[F5](../roadmap.md#f-localize-the-shortfall-before-changing-anything)'s question and not this
one's. This is the same move F2 made on clipping, for the same reason: a first-divergence hunt
confounded by a known convention difference locates the convention rather than the defect.

## The seven gates, and why they are ordered

The two routes stop being comparable step by step the moment they take a different equation,
so a naive *"walk both streams until an array differs"* would report the first array
difference and call it the divergence — which on these two implementations is a difference the
route already explains. The gates are ordered by the order in which a difference can first
bite; each is a self-contained comparison of a quantity both sides genuinely have; and the
**earliest failing one is what gets classified**.

| gate | question | class a failure belongs to |
| --- | --- | --- |
| 0 | did the two sides read the same numbers? | `input` |
| 1 | do `Q_r`, `g_{r,1}` and `g_{r,2}` at the **initial** pair agree? | `learner` |
| 2 | does a round take the same equations in the same order? | `update-order` |
| 3 | which reductions does each refit of a round contribute? | `reduction-vintage` |
| 4 | did both sides stop on a tolerance rather than on a cap? | `stopping-rule` |
| 5 | does this package's closing pass move the state R never takes? | `frozen-close` |
| 6 | do the reported `psi` and `se` agree? | `corrected-ic` |

Every gate downstream of the first failure is reported and marked **`confounded`**. That is
not a third verdict on the comparison — the numbers are what they are — but on what may be
*read off* it. Dropping them would leave a reader unable to see how far apart the two ended
up; printing them as findings is the mistake the ordering exists to prevent.

**Gate 1 is F3's own stopping rule** — *"stop immediately if the trace inputs or the first
reduced fits do not agree"* — and it cannot be read off either trajectory. R primes its loop
with a `Qr` refit and this package primes with an equation-(8) solve, so the first reduction
each *stream* records is taken at a different outcome regression. The comparable object is the
reduction at the initial pair, which is neither side's first step, so both sides compute and
export it on purpose.

**Gate 4 fails when R reached `maxIter`.** `drtmle`'s default is `maxIter = 3`, which is a
budget and not a tolerance; a run that reached it has not been compared on how it *stops* at
all. Reporting that as agreement would put a fact about the dispatch into a column about the
estimator.

**Gate 5 is not a comparison**, because the R package's loop has no analogue of
`_close_at_frozen_reductions` to compare against. What it reports is how far the closing pass
moves the state it was handed, which is the number that says whether the absence of an
analogue could matter.

## What the route comparison establishes, with no R installed

R's round is read off the published source and transcribed once, into `R_ROUND` in the test
module: `fluctuateG`, refit `gr`, `fluctuateQ2`, `fluctuateQ1`, refit `Qr`. Against it, this
package's two update orders read:

| | round 1, in order | equations | vintages adopted |
| --- | --- | --- | --- |
| `drtmle` (R) | `9 → refit gr → 10 → 8 → refit Qr` | `9, 10, 8` | **two** — `gr`, then `Qr` |
| `cleverly` | `9 → refit all → 10 → 8 → refit all` | `9, 10, 8` | **one** |
| `paper` | `8 → refit gr → 10 → refit Qr → 9` | `8, 10, 9` | **two** — `gr`, then `Qr` |

**This package's default takes R's equations in R's order and adopts one reduction vintage per
round. Its `"paper"` order adopts R's two vintages and takes the equations in a different
order. Neither is R's round.** That is a sharper statement than *"the routes differ"*, it is
the second, third and fourth rows of
[R3's table](../roadmap.md#f-localize-the-shortfall-before-changing-anything) read off a run
rather than off a reading of two sources, and it is what
[F4](../roadmap.md#f-localize-the-shortfall-before-changing-anything)'s ablation is handed: the
R-style trajectory F4 runs as a benchmark path is neither of the two constructions this package
already has.

It is pinned in `test_neither_order_is_r_s_round` so that a construction change has to move it,
and it needs no R to check, because the R half of it is the published source rather than a run.

**One thing the table does *not* say**, and the distinction is the whole of gate 2's ordering:
that `"cleverly"` matches R on the equations is not evidence that the equations are right. Both
descend from one source. It is evidence about a **transcription**, which is exactly the class
the standing refusal's reasoning covers and exactly what
[item 21](../roadmap.md#what-is-still-open) is the worked example of.

## What the run measured

`drtmle` 1.1.2 against this revision, on the frozen fixture, `n = 200` over the three committed
folds, `tolg = 0.01`. Both of R's outcome-update routes were run: its default `Qsteps = 2`
(`fluctuateQ2` then `fluctuateQ1`) and `Qsteps = 1` (the joint two-column solve). The wrappers
are exactly transparent on both — `verify = 0`, at a bar of `1e-12`.

**F3's stopping rule is cleared.** The two sides read the same numbers bit for bit, and their
first reduced fits agree at `3.69e-10` against a `1e-8` bar declared before any R number was
read. So everything downstream is readable as a *construction* difference rather than as an
input or a learner one — which is the thing this run had to establish before it could say
anything at all.

| gate | reading (`Qsteps = 2`) | verdict |
| --- | --- | --- |
| 0 inputs | bit for bit | **agree** |
| 1 first reduced fit | worst `3.69e-10` on `g_{r,1}`; `Q_r` and `g_{r,2}` at `1.1e-15` | **agree** |
| 2 update order | `R = 9 → refit gr → 10 → 8 → refit Qr` against `cleverly = 9 → refit all → 10 → 8 → refit all` and `paper = 8 → refit gr → 10 → refit Qr → 9` | **differ** |
| 3 reduction vintage | `R = gr+qr`; `cleverly = all+all`, `paper = gr+qr` | *confounded* — matches `paper` |
| 4 stopping rule | R: 2 rounds, cap 3, `tolIC = 0.005`; this package: 14 and 12 rounds, both on tolerance | *confounded* |
| 5 frozen close | the closing pass moves `Q*` by `5.0e-05` | *confounded* |
| 6 reported estimate | worst `0.0112`, on `se[ey1]` | *confounded* |

**The earliest divergence is gate 2, classified `update-order`.**

### The four readings

**1. The reduced regressions are the same objects, and that is now measured rather than
assumed.** `Q_r` and `g_{r,2}` agree at `1.1e-15` and `1.2e-15` — machine precision — which
says the designs, the targets, the fitting rows and the cross-fitting all line up across the
language boundary. `g_{r,1}` is the only one carrying more, at `3.69e-10`, and what that is
turned out not to be what it looked like: see *the two things the harness got wrong* below.

**2. Neither of this package's update orders is R's round, and they miss it in different
places.** `"cleverly"` takes R's equations in R's order — `9, 10, 8` — and adopts one
reduction vintage per round where R adopts two. `"paper"` adopts R's two vintages, `gr` then
`Qr`, and takes the equations in a different order — `8, 10, 9`. Every one of the three
implementations agrees that equation (10) sits between the other two; what they disagree about
is which of (8) and (9) opens the round, and whether a refit contributes one field group or
three. That is
[R3's second, third and fourth rows](../roadmap.md#f-localize-the-shortfall-before-changing-anything)
read off a run rather than off a reading of two sources, and it is what
[F4](../roadmap.md#f-localize-the-shortfall-before-changing-anything)'s ablation is handed: the
R-style trajectory F4 runs as a benchmark path is **neither** construction this package already
has, so it is a third arm rather than a relabelling of `update_order="paper"`.

**3. The two implementations do not stop at the same bar, and the gap is large.** `drtmle`'s
`tolIC` defaults to `1/n`, which is `0.005` here; it exits after **2** rounds at `Qsteps = 2`
and **1** at `Qsteps = 1`, both inside its `maxIter = 3` cap. This package exits on its own
tolerance after **14** rounds under `"cleverly"` and **12** under `"paper"`. So "R converges
faster" is not the fact — the fact is that the two declare convergence at different bars, and
a comparison of final states between them is in part a comparison of `1/n` against this
package's threshold. Whether driving the three scores below `1/n` is what Theorem 1 needs is a
question for the derivation and the remainder decomposition, and it is exactly the kind of
question this run exists to raise rather than to answer.

**4. The reported `se` differs by more than the reported `psi` does, and by a lot.** `psi[ate]`
is `+0.2179` in R against `+0.2175` under `"cleverly"` — `0.0005`, well under a hundredth of a
standard error. `se[ey1]` is `0.0491` against `0.0603`, a **23%** difference in the quantity the
whole variant exists to produce. That is consistent with what this package already knows about
itself — the extra equations cannot move `psi` and only move its variance — and it is
`confounded` here, because two implementations that took different routes to different fixed
points at different tolerances have no reason to report the same `se`. It is recorded because
the size of it is what makes gate 2 worth acting on rather than noting.

### What this does **not** establish

That `"cleverly"` matches R on the equations is **not** evidence that the equations are right.
Both descend from one source, so agreement between them is evidence about a **transcription**
— which is exactly the class the standing refusal's reasoning covers, and exactly what
[item 21](../roadmap.md#what-is-still-open) is the worked example of: a parity run would have
recorded R's sign as correct and been right by luck. Nothing here selects a construction. F4
measures the three constructions against `√n R_remaining`, the score means and the score-failure
rate; [F7](../roadmap.md#f-localize-the-shortfall-before-changing-anything) is the only row that
may change one, and only against the theorem and the remainder identities.

### One thing the run found about the R package itself

`drtmle` 1.1.2 documents `cvFolds` as accepting "a vector of fold assignments", and its
`make_validRows` implements exactly that. **The exported `drtmle()` cannot pass one**: it guards
with `if (cvFolds > 1)`, which raises `the condition has length > 1` on any vector, before
`make_validRows` is reached. So the documented path is unreachable from the exported function,
and the harness reaches it by wrapping `make_validRows` — the package's own supported branch,
called with the committed column. Recorded because it is why the fold wrapper exists, and
because a reader who tried `cvFolds = folds` would otherwise conclude the fixture was at fault.

## What the tests guard

`tests/unit/test_drtmle_r_compare.py`, 17 tests, ~3 s, two `DRTMLE` fits shared across every
one of them. **No R is installed and none is invoked** — what is tested is the half of F3 that
is Python.

- **the reader fails closed** — a truncated blob, a partial input file and a missing export
  each raise, with a message that says to rerun the R side rather than interpret the result;
- **the blob round-trips exactly** — `==`, not `allclose`, so a format that started rounding
  fails rather than passing at `1e-15`;
- **gate 0 has no tolerance** — one unit in the last place fails it, and is classified `input`;
- **each gate can fail, and gate 2 can also pass** — a synthetic export is built from a real
  trace so every gate passes by construction, and each test then breaks exactly one thing;
  without the passing case the failure tests would prove only that a broken gate fails;
- **only gates after the first failure are confounded**, and the failing gate is itself still
  read;
- **the report carries the refusal** — a reader arriving from a CI artefact has not read
  `CLAUDE.md`, and a table of divergences with no statement of what one *is* reads as a list of
  bugs in whichever implementation the reader trusts less.

## Three things the harness got wrong first, kept because they are the class of error F3 must not make

**Labelling a refit by what moved across it.** A refit step's `after` is its closure's whole
output; the round then adopts one field group from it or all three. Labelling the step by the
arrays that changed therefore reads `all` on **both** update orders and makes the vintage —
R3's fourth row, and the one difference no fitted result carries — invisible, while looking
like a working comparison. `benchmarks/drtmle_trace.vintages` is the instrument that reads
*adoption* rather than production, and it is what the route builder calls. The same mistake
appeared a second time in gate 3, which counted refit *steps*: that reads `2` for R, `2` for
`"cleverly"` and `2` for `"paper"` and calls all three the same. It compares the pattern now.

**Comparing the two states column for column when their arm axes differ.** `drtmle`'s `a_0` is
`(1, 0)` and a `Trace`'s `arms` is `(0, 1)`. A state built in the exporter's order and compared
against one built in the trace's compares arm 1 against arm 0, and it reads as a **`0.577`**
disagreement on `g_{r,2}` against a bar of `1e-8` — a large, entirely plausible `learner`
verdict on an axis bug. Both sides label their columns by arm, so the fix is to align by the
label; `test_the_state_is_read_in_the_traces_arm_order` requires the two orderings to give
*different* answers, so it cannot pass by their happening to coincide.

**Reading a solver tolerance as a penalty.** `g_{r,1}` first missed by `8.6e-05`, which looks
exactly like `LogisticRegression`'s residual L2 against `stats::glm`'s unpenalised IRLS. It is
not: at `tol=1e-12` the same `C=1e6` reads `7.5e-09`, so the whole of it was scikit-learn's
**default `tol=1e-4`**. Sweeping the penalty then moves the reading non-monotonically —
`7.5e-09`, `3.7e-10`, `7.9e-10`, `8.1e-09` at `C` from `1e6` to `1e15` — which is solver noise
and not a bias. Had the first reading been accepted, F3 would have stopped at gate 1 and
reported a learner difference that does not exist. `REDUCTION_TOLERANCE` was not moved.

**And one the instrument caught rather than a reader.** The first run's verify step failed at
`worst |traced - plain| = 0.00342` against `1e-12`. The wrappers were not the cause: the
*unwrapped* comparison fit was drawing its own random fold split, so it differed from the traced
one by its folds and not by the tracing. The fold wrapper is configuration rather than
recording, and is installed on every fit. That is what an instrument check is for, and it is
why the R script has one.
