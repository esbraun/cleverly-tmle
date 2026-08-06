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
| 5 | how near zero are equations (9) and (10) when each side stops? | `stopping-rule` |
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

**3. The two implementations declare convergence at bars seven orders of magnitude apart.**
This is the gate the first version of this comparison could not read, because the R side
exported no scores. `drtmle`'s `tolIC` defaults to `1/n` — `0.005` here — and it exits with
equations (9) and (10) at `1.64e-03`, inside its `maxIter = 3` cap. This package exits with them
at `7.94e-11`. Theorem 1's premise is about those empirical means, so *how near zero* is not a
detail of the loop: it is the premise, and the two implementations are not testing the same one.
Whether `1/n` is enough is a question for the derivation and the remainder decomposition, and it
is exactly what this run exists to raise rather than answer.

**4. The `se` gap is in the correction arrays, and specifically in one of them.** `psi[ate]` is
`+0.2179` in R against `+0.2175` under `"cleverly"` — under a hundredth of a standard error —
while `se[ey1]` is `0.0491` against `0.0603`, **23%**. Gate 7 says where that comes from:
`D*_Q[1]`'s spread is `0.3407` here against R's `0.0599`, a factor of `5.7`, while `D*_Q[0]`,
`D*_g[0]` and `D*_g[1]` are all within a factor of two. So the variance difference is not spread
across the curve — it is concentrated in equation (10)'s correction at the treated arm.
`confounded`, because two implementations at different fixed points reached at different bars
have no reason to agree; recorded, because its *shape* is what makes gate 3 worth acting on.

**And the signs agree throughout on `v1`.** The paper's display defines `D_A = -(Q_r/g)(A - g)`
while Theorem 1 *subtracts* `D_A`, and [item 21](../roadmap.md#what-is-still-open) adjudicated
that against the source's own appendices and resolved it **in favour of this package's positive
correction**. The R *code* carries the positive form too. That is worth one sentence and no
more: it is agreement about a transcription between two things descended from one source, which
is precisely the evidence item 21 says cannot settle the question — and did not settle it.

**5. R evaluates equation (10)'s block at the *initial* mechanism.** Inside R's loop
`eval_Dstar_Q` is handed `gn = gn` while `eval_Dstar_g` is handed `gn = gnStar`; the record
carries `at_targeted_g` on every block row and it reads `FALSE` on every `D_Q` and `TRUE` on
every `D_g`. Whether that is deliberate is a question for the derivation, and it is the class of
thing [item 20](../roadmap.md#what-is-still-open) turned on here: a block evaluated at one
mechanism while its equation was solved at another leaves the curve uncentred exactly where the
two differ. Recorded, not acted on.

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
