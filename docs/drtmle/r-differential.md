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
| `tests/unit/test_drtmle_r_compare.py` | 17 tests, ~3 s, **no R installed and none invoked** |

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

**The folds are handed over, not redrawn.** `drtmle`'s `cvFolds` accepts a *vector* of fold
assignments as well as a count, so R is given the fixture's committed `fold` column. Two
independent random splits would make every reduced regression differ at gate 1 and end the
comparison there, on a difference nobody was asking about.

**The reduced learner is a bare unpenalised GLM on both sides.** The frozen trace's
`reduced_*_learner="glm"` is a **two-candidate Super Learner** over `{mean, glm}`, and
`stats::glm` is one unpenalised fit; a convex combination against a single fit is a learner
difference this run already knows about. So both sides are given the bare GLM —
`LinearRegression` is `stats::glm(family = gaussian)` exactly, `LogisticRegression(C=1e6)` is
the binomial one to solver tolerance — matching `glm_Qr = "gn"` and `glm_gr = "Qn"` term for
term. What the *shipped* reduced learner does instead is
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
| 3 | how many refit vintages does one round adopt? | `reduction-vintage` |
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

<!-- F3-RUN-RESULTS -->
*Pending the first dispatch of `.github/workflows/drtmle-r-differential.yml`.* The gates, their
tolerances and the classification are frozen in this revision **before** any R numbers were
read, which is the same discipline
[the study manifest](study-manifest.md) applies and it matters more here than usual: a
threshold moved after seeing a comparison against another implementation is the failure mode
[stop-ship 17](../roadmap.md#stop-ship) names.

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

## One thing the harness got wrong first, kept here because it is the class of error F3 must not make

**Labelling a refit by what moved across it.** A refit step's `after` is its closure's whole
output; the round then adopts one field group from it or all three. Labelling the step by the
arrays that changed therefore reads `all` on **both** update orders and makes the vintage —
R3's fourth row, and the one difference no fitted result carries — invisible, while looking
like a working comparison. `benchmarks/drtmle_trace.vintages` is the instrument that reads
*adoption* rather than production, and it is what the route builder calls.

That was the first thing written here, and what caught it was a test asserting the `paper`
order's labels against F2's recorded reading of them. The general lesson is F2's own, in a
second place: **the state an equation used is read where its covariate is built, not where a
learner returned.**
