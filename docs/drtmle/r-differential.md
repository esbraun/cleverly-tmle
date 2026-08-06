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
| `benchmarks/r/drtmle_reference.R` | the R side: digests checked, `drtmle`'s own loop run with its internals wrapped, the step stream, the three influence-curve blocks and the whole state exported |
| `benchmarks/fixtures/r-trace-{v1,v2}-q{1,2}/` | **the committed records** — gzipped float64 with a SHA-256 manifest, so the comparison reproduces with **no R installed** |
| `benchmarks/drtmle_r_compare.py` | the Python side: the reader, the nine gates, the classification, the report |
| `.github/workflows/drtmle-r-differential.yml` | where R lives — dispatch only, nowhere else in the repository |
| `tests/unit/test_drtmle_r_compare.py` | 20 tests, ~7 s, **no R installed and none invoked** |

```bash
python -m benchmarks.drtmle_r_compare --fixture-version v1 --qsteps 2   # reads the record
Rscript benchmarks/r/drtmle_reference.R --out benchmarks/results/r-trace-v1-q2 --qsteps 2
```

**The records are committed, and a committed R record is a diagnostic record, not a truth.**
`CLAUDE.md` fences that by name: it may be read by `benchmarks/` and by tests that check an
*instrument is what it claims to be*, and **never** by a test asserting this package's `psi`,
`se` or curve agrees with it. What it buys is that the toolchain — ~25 minutes to build, `np`
and `crs` compiling NOMAD from source, and CRAN not always reachable — is needed once rather
than every time a question is asked. Verified the only way that means anything: with `Rscript`
moved off the path, the comparison runs and produces the tables below.

## Five choices that are load-bearing, and each is one F2 already had to make

**The package's own loop runs; nothing is re-implemented.** The R script replaces `drtmle`'s
internals with wrappers that call the originals and record either side of them, and restores
them in `on.exit` — the R idiom for exactly what `TracingDRTMLE` does in Python. A replay of
the loop written in the harness would be a second implementation, and a first-divergence hunt
whose instrument is a re-implementation finds the instrument. That the wrapping does not move
the fit is checked rather than argued: the script refits with the wrappers off and compares
`psi` and `cov` at `1e-12`, **fails closed**, and reads `0` on all four runs.

**Arrays cross the language boundary as raw little-endian float64.** F2's own record is why:
written at 17 significant digits and read back with a fast parser, the fixture's `w1` came back
short by one unit in the last place on 65 of 200 rows, at `2.2e-16` — precisely the size of
difference this run would find and mis-classify as a learner difference. Gzip is a *container*
and not a format, so the committed records keep that exactness while halving on disk.

**The folds are handed over, not redrawn.** Two independent random splits would make every
reduced regression differ at gate 2 and end the comparison there, on a difference nobody was
asking about. `drtmle`'s `make_validRows` accepts a *vector* of fold assignments as well as a
count, so R is given the fixture's committed `fold` column — reached by wrapping that function,
because the exported `drtmle()` cannot pass a vector through (see
[below](#one-thing-the-run-found-about-the-r-package-itself)).

**The reduced learner is a bare unpenalised GLM on both sides.** The frozen trace's
`reduced_*_learner="glm"` is a **two-candidate Super Learner** over `{mean, glm}`, and
`stats::glm` is one unpenalised fit; a convex combination against a single fit is a learner
difference this run already knows about. So both sides get the bare GLM — `LinearRegression`
is `stats::glm(family = gaussian)` exactly and agrees with it at `1.1e-15`; `LogisticRegression`
is the binomial one once its solver is actually run to convergence, at `3.7e-10`. What the
*shipped* reduced learner does instead is
[F5](../roadmap.md#f-localize-the-shortfall-before-changing-anything)'s question and not this
one's.

**Two fixtures, because one of them cannot ask the truncation question.** `v1`'s bound binds on
no row, deliberately — a first-divergence hunt confounded by a known convention difference
locates the convention. `v2` is `v1` with the mechanism strengthened until the bound bites (54
of 200 rows) and the bound tightened to meet it, and **nothing else moved**: the same draw, the
same truth, the same misspecified outcome regression. It is a *second file*, per F2's rule, not
an edit.

## The nine gates, and why they are ordered

The two routes stop being comparable step by step the moment they take a different equation,
so a naive *"walk both streams until an array differs"* would report the first array
difference and call it the divergence — which on these two implementations is a difference the
route already explains. The gates are ordered by the order in which a difference can first
bite; each is a self-contained comparison of a quantity both sides genuinely have; and the
**earliest failing one is what gets classified**.

| gate | question | class a failure belongs to |
| --- | --- | --- |
| 0 | did the two sides read the same numbers? | `input` |
| 1 | do the two truncation conventions give the same mechanism to divide by? | `truncation-convention` |
| 2 | do `Q_r`, `g_{r,1}` and `g_{r,2}` at the **initial** pair agree? | `learner` |
| 3 | does a round take the same equations in the same order? | `update-order` |
| 4 | which reductions does each refit of a round contribute? | `reduction-vintage` |
| 5 | did R solve equations (9) and (10) to the bar this package stops at? | `stopping-rule` |
| 6 | does this package's closing pass move the state R never takes? | `frozen-close` |
| 7 | do the two correction blocks agree row by row? | `corrected-ic` |
| 8 | do the reported `psi` and `se` agree? | `corrected-ic` |

Every gate carries an **absolute** and a **scale-relative** reading, as F3's row requires:
`5e-05` on an array whose `sd` is `0.6` is a different fact from the same number on one whose
`sd` is `5e-05`. Gates comparing a route or a vintage pattern have no scale and report `nan`
rather than a misleading `0`.

Every gate downstream of the first failure is reported and marked **`confounded`**. That is
not a third verdict on the comparison — the numbers are what they are — but on what may be
*read off* it. Dropping them would leave a reader unable to see how far apart the two ended
up; printing them as findings is the mistake the ordering exists to prevent.

**Gate 1 precedes gate 2, and the reason is causal.** This package forms `g_{r,2}`'s *target*
at the **truncated** mechanism — `reduced.py`'s `_roles` builds `(indicator - truncated) /
truncated`, and that module's docstring says why the bound is chosen at fit time here and
nowhere else — while `estimategrn` forms it at the untruncated `train_g`. So a truncation
difference does not wait for the targeting step to show up: it is already in what the reduced
regressions were asked to learn. Ordered the other way round it reads as a `learner` divergence
of `7.87`, which is true of the fitted values and wrong about the cause.

**Gate 2 is F3's own stopping rule** — *"stop immediately if the trace inputs or the first
reduced fits do not agree"* — and it cannot be read off either trajectory. R primes its loop
with a `Qr` refit and this package primes with an equation-(8) solve, so the first reduction
each *stream* records is taken at a different outcome regression. The comparable object is the
reduction at the initial pair, which is neither side's first step, so both sides compute and
export it on purpose.

**Gate 5's bar is this package's own, imported rather than restated**, and F3-closeout is where
that became true. It asks whether R solved its equations to the bar the loop here stops at —
`_solved`'s absolute clause, `_negligible_bar(n) = 1e-3/n`, which is `5e-6` at `n = 200` — and it
calls `cleverly.estimators.targeting`'s own functions to ask it. The predicate it replaced was
`R ≤ 10 × this package's *achieved* score` (`7.94e-11` on `v1`, so an effective bar of `7.9e-10`),
which is none of the three quantities in the question and is the bar/achievement conflation
[the ladder](#the-stopping-bar-ladder) exists to keep apart. **No reading moved**: the gate reads
`differ` on all four records before and after, and the earliest divergence is unchanged on each.

One clause is not evaluable across the boundary and the reading says so rather than inventing it.
`_solved` is `relative <= 1e-10` **or** `absolute <= 5e-6`; the relative clause divides by
`score_scale`, which the R export does not carry — `blocks.csv` has `mean` and `sd`, and `sd` is
not that scale. So the gate applies the absolute clause exactly and prints this package's achieved
figure beside it. That errs toward reporting a difference, never toward passing a run that solved
its equations less tightly than this package requires.

**Gate 6 is not a comparison**, because the R package's loop has no analogue of
`_close_at_frozen_reductions`. What it reports is how far the closing pass moves the state it
was handed, which is the number that says whether the absence of an analogue could matter.

## What the run measured

`drtmle` 1.1.2, both fixtures, both of R's outcome-update routes — its default `Qsteps = 2`
(`fluctuateQ2` then `fluctuateQ1`) and `Qsteps = 1` (the joint two-column solve). Four runs,
`verify = 0` on every one.

### `v1` — the truncation is slack, and the earliest divergence is the update order

| gate | reading (`Qsteps = 2`) | abs | rel | verdict |
| --- | --- | --- | --- | --- |
| 0 inputs | bit for bit | `0` | `0` | **agree** |
| 1 truncation | the bound binds on 0/200 — vacuous, nothing clips | `0` | `0` | **agree** |
| 2 first reduced fit | worst on `g_{r,1}`; `Q_r` and `g_{r,2}` at `1.1e-15` | `3.69e-10` | `4.6e-09` | **agree** |
| 3 update order | R `9→refit gr→10→8→refit Qr`; `cleverly` `9→refit all→10→8→refit all`; `paper` `8→refit gr→10→refit Qr→9` | — | — | **differ** |
| 4 reduction vintage | R `gr+qr`; `cleverly` `all+all`, `paper` `gr+qr` | — | — | *confounded* |
| 5 exit scores | R `1.64e-03` after 2 rounds (cap 3, `tolIC = 0.005`); this package `7.94e-11` after 14 and 12 | `1.64e-03` | `2.1e+07` | *confounded* |
| 6 frozen close | the closing pass moves `Q*` by `5.0e-05` | `5.0e-05` | `3.5e-04` | *confounded* |
| 7 correction arrays | worst on `D*_Q[1]`, `sd` R `0.0599` against `0.3407`; **signs agree throughout** | `3.49` | `14.3` | *confounded* |
| 8 reported estimate | worst on `se[ey1]` | `0.0112` | `0.229` | *confounded* |

**Earliest divergence: gate 3, `update-order`.**

### `v2` — the truncation binds, and it bites first

| gate | reading (`Qsteps = 2`) | abs | rel | verdict |
| --- | --- | --- | --- | --- |
| 0 inputs | bit for bit | `0` | `0` | **agree** |
| 1 truncation | the bound binds on **54/200**; the conventions differ on all 54 | `0.141` | `0.554` | **differ** |
| 2 first reduced fit | worst on `g_{r,2}`, `sd` R `2.2829` against `0.5297` | `7.87` | `4.68` | *confounded* |
| 5 exit scores | R `2.14e-03` after 2 rounds; this package `2.26e-11` after 11 and 16 | `2.14e-03` | `9.5e+07` | *confounded* |
| 8 reported estimate | worst on `psi[ate]` | `0.0298` | `0.405` | *confounded* |

**Earliest divergence: gate 1, `truncation-convention`.**

### The five readings

**1. The reduced regressions are the same objects, and that is measured rather than assumed.**
On `v1`, `Q_r` and `g_{r,2}` agree at `1.1e-15` and `1.2e-15` — machine precision — which says
the designs, the targets, the fitting rows and the cross-fitting all line up across the language
boundary. That is what makes every difference below a *construction* difference.

**2. Neither of this package's update orders is R's round, and they miss it in different
places.** `"cleverly"` takes R's equations in R's order — `9, 10, 8` — and adopts one reduction
vintage per round where R adopts two. `"paper"` adopts R's two vintages, `gr` then `Qr`, and
takes the equations in a different order — `8, 10, 9`. All three agree that equation (10) sits
between the other two; they disagree about which of (8) and (9) opens the round. That is
[R3's second, third and fourth rows](../roadmap.md#f-localize-the-shortfall-before-changing-anything)
read off a run, and it is what
[F4](../roadmap.md#f-localize-the-shortfall-before-changing-anything)'s ablation is handed: the
R-style trajectory is a **third arm**, not a relabelling of `update_order="paper"`.

**3. The two stopping rules differ by 1000×, and the states they reach by 2×10⁷.** Those are
two facts and an earlier revision of this document merged them into one. Both packages render
the same `o_p(n^{-1/2})` condition as an absolute bar of the form `c/n` on the three empirical
means:

| | absolute bar at `n = 200` |
| --- | --- |
| `drtmle`, `tolIC = 1/n` | `5e-3` |
| this package, `_NEGLIGIBLE / n = 1e-3 / n` (`targeting.py:120`) | `5e-6` |

so the **rules** are three orders apart, not seven. This package then *overshoots* its own bar
by five further orders — it reaches `7.94e-11` — because `_solved` is
`relative <= 1e-10` **or** `absolute <= 5e-6` and the relative test keeps it iterating. R stops
at `1.64e-03`, just inside its own. The 2×10⁷ figure is the gap between the two **achieved**
states; the 1000× is the gap between the two **rules**.

**4. The `se` gap was mostly the stopping bar, and the ladder is how that is known.** Gate 7
found `D*_Q[1]`'s spread at `0.3407` here against R's `0.0599` and could not say whether that
was a construction difference or an artefact of R having stopped early. Running R down a
tolerance ladder splits it — see [the ladder](#the-stopping-bar-ladder) below. Short version:
**most of it was the bar** — `64%` to `92%` of it depending on how the question is asked, which
is a range rather than a number because the reading is not invariant, and the verdict is the
same under every reading in it.

**And the signs agree throughout on `v1`.** The paper's display defines `D_A = -(Q_r/g)(A - g)`
while Theorem 1 *subtracts* `D_A`, and [item 21](../roadmap.md#what-is-still-open) adjudicated
that against the source's own appendices and resolved it **in favour of this package's positive
correction**. The R *code* carries the positive form too. That is worth one sentence and no
more: it is agreement about a transcription between two things descended from one source, which
is precisely the evidence item 21 says cannot settle the question — and did not settle it.

**5. R *passes* equation (10)'s block the initial mechanism — and on this reduction the callee
does not read it.** Inside R's loop `eval_Dstar_Q` is handed `gn = gn` while `eval_Dstar_g` is
handed `gn = gnStar`; the record carries `at_targeted_g` on every block row, and on `v1` it reads
`FALSE` on every `D_Q` row from round 1 onward and `TRUE` on every `D_g` row. (The two `TRUE`
`D_Q` rows are the `prime` phase at round 0, where R has just set `gnStar <- gn` and the two
arrays *are* the same one — an earlier revision said "`FALSE` on every `D_Q`" and that was two
rows too many.) **That flag records a call site, not a read**, because
it is computed by comparing the *argument* against the current state
(`benchmarks/r/drtmle_reference.R`) — and the two are different questions wherever the callee
branches on something else.

Here it branches on the reduction. Every run in this document is `reduction = "univariate"`, and
[the concordance's §10](theorem-concordance.md#10-the-bivariate-construction) — written from the
R source before any of this ran, and repeated in
[the roadmap's `reduction="bivariate"` bullet](../roadmap.md#d-widen-the-scope-to-what-the-sources-derive)
— records that the initial `g`
enters `eval_Dstar_Q`'s **bivariate** branch, `1{A=a}/grn2 · (grn2 − g)/g · (Y − Q)`, and that
*on the univariate branch the argument is unused*. So there is no univariate block evaluated at
the wrong mechanism and nothing here is uncentred: this is a **bivariate-only** observation,
carried for whoever writes that reduction, and it is **not** a construction candidate for
[F4](../roadmap.md#f-localize-the-shortfall-before-changing-anything) or F7.

**An earlier revision of this section read it the other way**, and the correction is worth its
own sentence because of *what fixed it*. Nothing was rerun and no R was installed: two documents
in this repository disagreed, and the one written from the source at the point the bivariate
construction was scoped is the one that holds. That is the shape
[item 20](../roadmap.md#what-is-still-open) is the worked example of — a recorded number
readjudicated in-process against what the repository already knew — and it is the reason a
`FALSE` in a fixture is a fact about an argument until something says what consumes it.


## The stopping-bar ladder

Gate 7's finding was recorded and **not interpretable**: two implementations at different fixed
points reached at different bars have no reason to agree, so a `5.7×` difference in
`sd(D*_Q[1])` could have been a construction difference or could have been R stopping early.
One knob splits them. `drtmle` takes `tolIC`, so R was run down a ladder with `maxIter` raised
to 100, and **it converges at every rung** — no cap reached, the achieved score falling
monotonically with the bar.

| `tolIC` | rounds | worst `P_n D` | `sd(D*_Q[1])` | `psi[ate]` | `se[ey1]` |
| --- | --- | --- | --- | --- | --- |
| `5e-3` — R's own default `1/n` | 2 | `1.64e-03` | `0.0601` | `+0.217908` | `0.049095` |
| `5e-6` — **this package's bar** | 10 | `4.78e-06` | `0.2693` | `+0.210142` | `0.057384` |
| `1e-8` | 17 | `4.48e-09` | `0.2692` | `+0.210133` | `0.057377` |
| `1e-10` | 21 | `8.29e-11` | `0.2692` | `+0.210133` | `0.057377` |
| *this package,* `cleverly` | 14 | `7.94e-11` | `0.3407` | `+0.217455` | `0.060323` |
| *this package,* `paper` | 12 | — | `0.2140` | `+0.215188` | `0.053918` |

**Verdict: `partial`** — and the thresholds were declared in the module before the first rung
was read, which is why it says `partial` rather than being rounded to `closed`. `closed` needed
both a spread ratio inside `1.2` and an `se[ey1]` ratio inside `1.05`. Read per order, since a
ratio and a percentage taken against different orders are not a comparison of anything:
`cleverly` reads `1.266` and `1.051`, `paper` reads `0.795` and `0.940`. Every one of the four
misses, narrowly, and the two orders miss on *opposite sides* of agreement — which is the same
route evidence the next paragraph reads, arriving here first.

**And "how much of the gap the bar explains" is not one number.** `sd(D*_Q[1])` moves `0.0601` →
`0.2692` against `paper`'s `0.2140` and `cleverly`'s `0.3407`. On the ratio the module computes —
this package over R, distance from agreement — that is **92%**; with the ratio reversed it is
**64%**; and on the raw absolute gap, the one reading with no orientation to choose, **64%**. An
earlier revision printed the `92%` alone and beside `cleverly`'s two ratios, which was two
mistakes in one sentence: a non-invariant statistic reported as though it were invariant, and
two orders read as one. **The verdict does not turn on any of it** — `persists` needed under
half explained and all three readings clear it — so `partial` is robust to the choice rather than
an artefact of it, and that is now shown rather than assumed. No threshold moved.

Three things follow, and the second is the one worth carrying forward.

**Most of the gap was the bar.** `sd(D*_Q[1])` moves from `0.0601` to `0.2692` — a factor of
`4.5` — as soon as R is asked to solve its equations as tightly as this package does, and it is
stable across three further orders of magnitude after that. What gate 7 measured was mostly
*"R stopped after two rounds"*, not a defect in either corrected influence curve. The document's
earlier framing of it as a localized `se` difference overstated what was known; *"overwhelmingly"*
then overstated it in the other direction, on the `92%` reading alone.

**R's converged state lands *between* this package's two update orders.** `sd(D*_Q[1])`:
`paper` `0.2140`, **R `0.2692`**, `cleverly` `0.3407`. `se[ey1]`: `paper` `0.0539`, **R
`0.0574`**, `cleverly` `0.0603`. So the residue after the bar is accounted for is *route*, not
construction — which is what makes it F4's question rather than a separate one. Gate 7 now
reports per order for exactly this reason; its worst-across-orders reading could not see it.

**And three converged solutions sit at three different points.** All of `cleverly`, `paper` and
R at `tolIC = 1e-10` drive the three empirical means to `1e-10` or better, and their `psi[ate]`
are `+0.217455`, `+0.215188` and `+0.210133` — spread over about a tenth of a standard error.
[Item 22](../roadmap.md#what-is-still-open) asks whether the two routes reach the same fixed
point on real data; this is a third implementation saying the fixed point is **route-dependent**,
on a draw where all three converged. That is evidence for the item rather than an answer to it —
one draw, `n = 200` — and it is F4's to test at scale.

### One thing this says about `drtmle`'s shipped defaults

At `maxIter = 3` and `tolIC = 1/n`, R exits after 2 rounds having *not* reached its own fixed
point: `psi[ate]` moves from `+0.217908` to `+0.210133` between there and convergence, about
`0.10` standard errors, and `se[ey1]` by 17%. Recorded as a fact about a configuration, not as
a criticism — a default is a trade against runtime, and nothing here establishes which point is
the right one to report. It is noted because a reader comparing against `drtmle` at its defaults
is comparing against an unconverged alternation, which is not what either package's derivation
is about.

### What `v2` adds, and it is the sharpest thing here

**The two truncation conventions cannot be reconciled, and the difference reaches the reduced
regressions before it reaches anything else.** `drtmle`'s `tolg` is a scalar **lower** bound
applied to each arm's `g` independently; this package's `g_bounds` is a pair and
`Propensity.bounded` clips `g_1` and takes the complement. With two arms a row clipped low on
one arm is clipped *high* on the other, so no choice of bound arranges them into agreement — the
difference is a fact about the two conventions rather than a confounder the fixture failed to
remove.

What it costs is not small. `g_{r,2}`'s target is `(1{A=a} - g)/g`, and this package forms it at
the **truncated** mechanism while `estimategrn` forms it at the untruncated one. At a row where
`g = 0.0089`, the untruncated target for a treated unit is `111`; at the truncated `0.15` it is
`5.67`. Fitted, the two `g_{r,2}` arrays have spreads of `2.28` and `0.53` — a factor of `4.3` —
and differ by `7.87` at the worst row. That is one line of `reduced.py` against one line of
`estimate.R`, and it is the most precisely localized difference this run produced.

**It is also the one where this package has already written down a reason.** `fit_reduced`'s
docstring says the bound is chosen at fit time here and nowhere else, because `g_{r,2}`'s target
*is* a quotient by the mechanism and cannot be left raw and re-truncated later — and records the
consequence, that `truncation_curve` is flat by construction on that part of the curve. Whether
that reasoning or R's is what Theorem 1 needs is F7's question, adjudicated against the
derivation. Nothing here decides it.

### One thing the run found about the R package itself

`drtmle` 1.1.2 documents `cvFolds` as accepting "a vector of fold assignments", and its
`make_validRows` implements exactly that. **The exported `drtmle()` cannot pass one**: it guards
with `if (cvFolds > 1)`, which raises `the condition has length > 1` on any vector, before
`make_validRows` is reached. So the documented path is unreachable from the exported function,
and the harness reaches it by wrapping that function — the package's own supported branch,
called with the committed column. Recorded because it is why the fold wrapper exists, and
because a reader who tried `cvFolds = folds` would otherwise conclude the fixture was at fault.

## What the tests guard

`tests/unit/test_drtmle_r_compare.py`, 20 tests, ~7 s, two `DRTMLE` fits shared across all of
them. **No R is installed and none is invoked** — and they read the *committed records* rather
than a synthetic stand-in, so the reader is exercised against the artefact it will meet and
gates 0 to 2 agree because two implementations genuinely agree there.

- **the reader fails closed** — a bad digest, a missing manifest, a truncated blob and a partial
  input file each raise, with a message that says to rerun the R side rather than interpret;
- **gate 0 has no tolerance** — one unit in the last place fails it, classified `input`;
- **each gate can fail** — a copy of the record is mutated, one thing at a time, and the manifest
  recomputed so the gate fails rather than the digest check;
- **only gates after the first failure are confounded**, and the failing gate is itself read;
- **the ordering is asserted** — truncation before the reduced fit, with the reason;
- **the committed record is R's round**, which is an *instrument-validity* check and not a
  correctness one: it asks whether the record says what this repository believes `drtmle`'s loop
  does, and asserts nothing about `psi`, `se` or any curve;
- **the report carries the refusal** — a reader arriving from a CI artefact has not read
  `CLAUDE.md`, and a table of divergences with no statement of what one *is* reads as a list of
  bugs in whichever implementation the reader trusts less.

`tests/unit/test_drtmle_trace.py` gains the mirror set for `v2`: that it regenerates from its
seed, that the bound binds materially rather than on a row or two, that **only the mechanism
moved** against `v1`, and that it is not degenerate — a `v2` that tidied the outcome regression
while strengthening the mechanism would clip beautifully and see nothing.

## Four things the harness got wrong first, kept because they are the class of error F3 must not make

**Labelling a refit by what moved across it.** A refit step's `after` is its closure's whole
output; the round then adopts one field group from it or all three. Labelling the step by the
arrays that changed reads `all` on **both** update orders and makes the vintage — R3's fourth
row, and the one difference no fitted result carries — invisible, while looking like a working
comparison. `benchmarks/drtmle_trace.vintages` reads *adoption* rather than production. The same
mistake appeared a second time in the vintage gate, which counted refit *steps*: that reads `2`
for R, `2` for `"cleverly"` and `2` for `"paper"` and calls all three the same. It compares the
pattern now.

**Comparing the two states column for column when their arm axes differ.** `drtmle`'s `a_0` is
`(1, 0)` and a `Trace`'s `arms` is `(0, 1)`. A state built in the exporter's order and compared
against one built in the trace's compares arm 1 against arm 0, and it reads as a **`0.577`**
disagreement on `g_{r,2}` — a large, entirely plausible `learner` verdict on an axis bug. Both
sides label their columns by arm, so the fix is to align by the label.

**Reading a solver tolerance as a penalty.** `g_{r,1}` first missed by `8.6e-05`, which looks
exactly like `LogisticRegression`'s residual L2 against `stats::glm`'s unpenalised IRLS. It is
not: at `tol=1e-12` the same `C=1e6` reads `7.5e-09`, so the whole of it was scikit-learn's
**default `tol=1e-4`**. Sweeping the penalty then moves the reading non-monotonically —
`7.5e-09`, `3.7e-10`, `7.9e-10`, `8.1e-09` at `C` from `1e6` to `1e15` — which is solver noise
and not a bias. Had the first reading been accepted, F3 would have stopped at gate 2 and reported
a learner difference that does not exist. `REDUCTION_TOLERANCE` was not moved.

**Calling a sign flip on arrays that are far apart either way.** The correction gate takes
whichever of `±theirs` is nearer, and on `v2` — where the two sides are at genuinely different
fixed points — that picked "flipped" on arrays whose distance is dominated by everything else.
Reporting that as *"the signs differ"* would manufacture exactly the finding item 21 warns
about, so a flip is now only read as one when negating is decisively better.

**And one the instrument caught rather than a reader.** The first run's verify step failed at
`worst |traced - plain| = 0.00342` against `1e-12`. The wrappers were not the cause: the
*unwrapped* comparison fit was drawing its own random fold split, so it differed from the traced
one by its folds and not by the tracing. The fold wrapper is configuration rather than recording,
and is installed on every fit. That is what an instrument check is for, and it is why the R
script has one.
