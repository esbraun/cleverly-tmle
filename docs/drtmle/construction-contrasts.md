# The construction contrasts, and what one factor at a time can be made to say

[The roadmap](../roadmap.md)'s [piece F](../roadmap.md#f-localize-the-shortfall-before-changing-anything)
puts **F4** fifth, beside F5, and this document is that instrument's record. F4 is the
**construction** half of the localization and F5 is the learner half; the two are independent
unresolved causes, and either remains a theorem premise even if the other succeeds.

**It closes nothing, and that is what it is for.** Nothing under `src/` moves here; only F7 may.
A difference this makes visible is a *question*, adjudicated against Theorem 1,
[the concordance](theorem-concordance.md), the exact-law identities and the remainder
decomposition — never settled by which side another implementation is on
([stop-ship 17](../roadmap.md#stop-ship)). **F4 may not branch, and a null result is a result.**
Final coverage is in neither diagnostic — that is F8's and only F8's.

## What it is

| piece | what it holds |
| --- | --- |
| `benchmarks/drtmle_construction.py` | the harness: the arms, the frozen rule, the two phases, the contrast arithmetic |
| `benchmarks/drtmle_trace.py` | `RStyleDRTMLE` — F4's **third arm**, and the `"r-style"` route the trace harness now takes |
| [`evidence/f4-construction/prereg.json`](../../evidence/f4-construction/prereg.json) | the frozen design, committed before the first decision fit |
| [`evidence/f4-construction/README.md`](../../evidence/f4-construction/README.md) | what each part of it is, and what a run does with it |
| [§9 of the validation plan](validation-plan.md#9-the-construction-contrasts-piece-f4) | the rule a verdict is read against |
| `.github/workflows/drtmle-construction.yml` | the dispatch: one job per `(cell, size)`, plus the truncation job |
| `scripts/recover_construction.sh` | the rows back out of a job log, digest-checked |
| `tests/unit/test_drtmle_construction.py` | 35 tests, ~32 s, **no R installed and none invoked** |

```bash
python -m benchmarks.drtmle_construction --phase prereg --out evidence/f4-construction
python -m benchmarks.drtmle_construction --phase truncation
python -m benchmarks.drtmle_construction --phase run \
    --prereg evidence/f4-construction/prereg.json --cohort selection
```

## The six factors, and why they had to be six

**A contrast that moves two things cannot say which of them moved the number.** F3 is what makes
that concrete rather than methodological: this package's two update orders differ from R's round
in *two* crossed ways — the equation order and how many reduction vintages a round adopts — so
"`cleverly` against an R-style arm" was never one factor.

| contrast | factor isolated | how the arm is built |
| --- | --- | --- |
| `r-style` ~ `cleverly` | reduction **vintage**, `all+all` against `gr+qr` | `RStyleDRTMLE` |
| `paper` ~ `r-style` | **equation order** and refit placement | `update_order="paper"` |
| `no-close` ~ `cleverly` | the frozen-reduction **closing pass** | scoped identity `_close_at_frozen_reductions` |
| `nested` ~ `cleverly` | the reduction **cross-fitting** construction | `reduced_crossfit="nested"` |
| `loose` ~ `cleverly` | the **stopping rule** | scoped `_NEGLIGIBLE = 1.0`, i.e. R's `tolIC = 1/n` |
| bounded ~ raw | the **truncation** convention | `raw_target`, read exactly — see below |

**Five of the six are cohort contrasts and the sixth is not**, which is a measurement rather than
a shortcut and is [its own section](#the-sixth-factor-is-answered-exactly-and-a-cohort-could-not-have-answered-it).

## Three choices that are load-bearing

**The R-style arm is a two-line adoption change and not a reimplemented loop.**
`targeting.solve_with_reduction`'s `"cleverly"` branch already runs R's equation order —
`9 → refit → 10 → 8 → refit` — with the two refits in R's own *placement*. What differs is that
both refits adopt the **whole** `ReducedSet` where R adopts `gr` from the first and `Qr` from the
second; and the `"paper"` branch already does a partial adoption, with the same
`dataclasses.replace` calls, under a *different* equation order. So the one factor separating
this package's round from R's is which fields a refit is allowed to contribute, and that is
reachable from the refit closure alone. The shipped loop is the loop that runs, which is what
makes the arm one factor rather than a second implementation whose drift from the first would be
invisible.

**Three arms are scoped patches of module-level names, because that is the only route there.**
`max_outer`, `_NEGLIGIBLE`, `_STALL_FACTOR` and the fact that the closing pass always runs are
constants in `src/` that no keyword reaches. The patches are installed for one fit and restored
in a `finally` — the route `TracingDRTMLE` already takes — so an ordinary `DRTMLE` in the same
process is untouched and a raise inside the alternation cannot leave the module patched. Both
halves are tested, the second by raising on purpose.

**The pre-close state is a *result* here and not a state.** F2's trace records the boundary and
`Trace.boundary()` returns it, but every column F4 reports comes off a fitted result — so the two
sides of that contrast have to be two results. The `no-close` arm replaces the closing pass with
an identity that hands the loop's exit state straight back, recomputing nothing: the loop has
already restated all three scores at the pair the round exits at, so the incoming record *is* the
pre-close record. It is checked against F2's recorded boundary rather than against its own
construction.

## What the instrument establishes so far

**These are readings from the harness and its frozen fixtures rather than from the decision
cohorts**, so they are exact and deterministic where the cohort table below is statistical. The
selection cohort's table follows them; the audit cohort's is what turns it into a verdict.

### 1. The R-style arm reproduces R's round exactly, and it is the first arm here that does

`('9', 'refit:gr', '10', '8', 'refit:qr')` — the identical tuple the committed R record carries.
Read against the four `r-trace-*` records through F3's own gates:

| gate | before F4 | with the third arm |
| --- | --- | --- |
| 3 update order | `differ` — *matches neither* | **`agree`** — matches `r-style` |
| 4 reduction vintage | `confounded` | **`agree`** — `r-style` and `paper` both `gr+qr` |

The two factors are **crossed**, which is what F3 established and what this pins:
`cleverly` takes R's equation order with one vintage (`all+all`), `paper` takes R's two vintages
under a different order, and only the third arm takes both of R's. The equation subsequence of
`r-style` is `cleverly`'s and its vintage pattern is `paper`'s.

**This is an instrument-validity reading and not a correctness one.** It asks whether the arm
labelled R-style takes the trajectory this repository says R takes, which is the premise every
contrast against it rests on. It asserts nothing about `psi`, `se` or any curve — the line
[`CLAUDE.md`](../../CLAUDE.md)'s fence draws, and the reason a committed R record may be read
here at all.

### 2. The truncation convention differs at the initial fit and is gone by convergence

Twelve deterministic rows, both frozen fixtures, both ends of the alternation:

| fixture | clipped | stage | `qr` | `gr1` | `gr2` |
| --- | --- | --- | --- | --- | --- |
| `v1` | 0/200 | initial | identical | identical | identical |
| `v1` | 0/200 | converged | identical | identical | identical |
| `v2` | 54/200 | initial | identical | identical | **differ, `7.28e-01`** |
| `v2` | 54/200 | converged | identical | identical | identical |

**The second row of `v2` is the finding.** The loop carries the **truncated** tilt forward —
item 20's fix, *"the truncated tilt, which is what makes the next round's offset, every later
covariate and the reported correction read one array"* — so from the first mechanism update `g*`
already lies inside the covariate bounds and re-truncating it for a reduction target is a no-op.
The convention therefore differs where it is *applied* and agrees everywhere the reported curve
reads. **No comparison of fitted results could have seen this**, in either direction: it would
have read `identical` and concluded the factor was inert, when in fact it is active and then
absorbed.

`v2` is bound-active, so [§7](theorem-concordance.md#7-truncation-is-not-in-the-theorems-algorithm)
puts it outside the theorem's scope and this carries no theorem-backed claim. It is recorded as
out-of-theorem behaviour.

### 3. Six draws is not a pilot, and the size is what makes the column noisy

Sizing this study needed the **paired** spread of each contrast, which no study here had
measured. Three things came out of measuring it, on a third seed stream disjoint from both
cohorts, and each was a hypothesis this page had to give up:

- **E1b's `0.33` is the wrong input.** It is the across-draw spread of a *single arm's* column;
  every outcome here is a paired difference. Measured, the paired spreads are five to twenty
  times larger than it at `n = 600` and an order of magnitude smaller at `n = 2,400`.
- **A six-draw pilot read three contrasts low by a factor of twenty** against a twelve-draw one
  (`0.0001`–`1.36` against `0.0001`–`2.67`). A spread estimated from six paired differences is
  not an estimate of a spread, and the first version of this design was sized from one.
- **The quadrature rule is not the driver**, which was checked rather than assumed because E1's
  whole subject makes it the obvious suspect. At 512 against 2,048 points the paired spreads move
  by under 3%. What drives it is the size: at `n = 600` the arms land at genuinely different
  fixed points draw by draw, and by `n = 2,400` they do not.

So the two sizes carry different committed counts and the manifest **declares** what each can
answer — `n = 2,400` resolves `moved` and `flat`, `n = 600` resolves `moved` only. That is a
stated limit of the study rather than a result of it.

### 4. The study is about a third of C3c and every number in its size is derived

| | |
| --- | --- |
| what the first design asked for | ~6,048 fits — *the size F4's row forbids by name* |
| what the evidence cut it to | **~2,496 fits** |
| where the cut came from | per-size counts from the measured paired spread; the truncation contrast moved off the cohort |

## The contrast table — the selection cohort

**The audit cohort is still running, and until it lands nothing below is a verdict.** The two
cohorts are disjoint sets of simulation draws and the second exists because an effect assessed
on the draws that produced it has not been reproduced. Read the audit rows beside these or read
neither.

208 draws, 6 arms, **1,248 fits, zero errors**; rows in
[`evidence/f4-construction/`](../../evidence/f4-construction/).

### The reference arm reproduces C3c's column, which is what says the harness measures the
### quantity the shortfall is about

| cell | `cleverly` here | C3c |
| --- | --- | --- |
| `q-drift` `n = 600` / `2,400` | `1.2030` / `1.1289` | `1.17`–`1.25` |
| `g-drift` `n = 600` / `2,400` | `3.8986` / `3.7315` | `4.3`–`4.83` |

### `√n R_remaining`, the primary column

| contrast | `q` 600 | `q` 2,400 | `g` 600 | `g` 2,400 |
| --- | --- | --- | --- | --- |
| `r-style ~ cleverly` | `flat` | *see the defect below* | `flat` | `unresolved` |
| `paper ~ r-style` | `unresolved` | `unresolved` | `unresolved` | `unresolved` |
| `no-close ~ cleverly` | `unresolved` | `unresolved` | `flat` | `flat` |
| `nested ~ cleverly` | **`moved` `−0.203`** | `unresolved` | `unresolved` | `unresolved` |
| `loose ~ cleverly` | `unresolved` | **`moved` `+0.219`** | **`moved` `−1.006`** | `unresolved` |

**Neither `moved` factor clears F4's acceptance, and the reason is in the table rather than in a
threshold.** That clause asks an effect to reproduce on the audit cohort **and at both sizes in
the affected regime**. `nested` fires in one cell of four. `loose` fires in two and **with
opposite signs** — it improves `g-drift` by `−1.006` and worsens `q-drift` by `+0.219`, which is
not one effect measured twice.

### The score-failure rate, and the one four-for-four reading

`no-close` moves it by `+3.93` to `+4.98` in **every** cell, with `valid = False` on all 208
draws — and moves `√n R_remaining` not at all (`−0.077` to `+0.051`, `flat` or `unresolved`).

**The frozen-reduction closing pass is what makes the reported scores solved, and it is not
touching the remainder.** That is an exoneration rather than a localization: the pass is
load-bearing, and a construction diagnostic that removed it would break score validity while
leaving the shortfall exactly where it was. Every other arm's score-failure contrast is `flat`
except `loose` at `g-drift` `n = 2,400` (`+0.375`).

### A defect in the frozen rule, found by the run and not repaired after it

`r-style ~ cleverly` at `q-drift` `n = 2,400` reads **`moved`** — on a mean of `−0.0000` with an
interval of `[−0.0001, −0.0000]`. That interval excludes zero **and** lies far inside the
negligible margin `±0.125`, so [§9](validation-plan.md#the-contrast-rule-frozen-before-the-dispatch)'s
clause 1 and clause 2 **both fire, and the frozen rule does not say which wins**.
`_verdict` tests `moved` first, so an effect of `4e-05` — `0.003%` of the column — is labelled a
localization.

**The margin is not being moved after seeing a result**, which is what
[stop-ship 17](../roadmap.md#stop-ship) forbids and what the freeze exists to prevent. The
ambiguity is recorded here, both readings are reported, and the substantive conclusion rests on
the **magnitude**: at `4e-05` against a column of `1.13`, the reduction vintage does nothing.
Whoever writes the next rule of this shape should order the clauses **`flat` before `moved`**,
which is ordinary equivalence-testing logic and is what the margin was declared for.

### What this says so far

**No construction factor moves the remainder consistently.** The largest effect anywhere is
`loose`'s `−1.006` in one cell, and it reverses sign in another; the largest that reproduces in
direction across sizes is nothing. Against a column standing at `1.13`–`3.90` that has to fall
to zero, and a coverage shortfall of `0.844` against `0.95`, **none of the six factors is a
candidate for the gap.** Both of F3's own nominations — the reduction vintage and the truncation
convention — are inert, one measured at `4e-05` and the other proved identical at convergence.

If the audit cohort reproduces this, F4's answer is a **null**, which its row admits as a result,
and the live hypotheses become the two F4 cannot test: whether the reduced regressions are
consistent (F5's question, still `unverified` in the concordance) and whether these fits meet
Theorem 1's expansion premise at these sizes at all.

## The sixth factor is answered exactly, and a cohort could not have answered it

**The declared bound-active design was not bound-active**, and finding that out is what moved
this contrast off the cohort. The tier-2 law is `linear_dgp`, chosen for overlap rather than for
difficulty, and its initial mechanism's clip share is `0.0000` **even at a bound of
`(0.15, 0.85)`** — so `raw` and `cleverly` are bit-identical on every draw this study takes. A
cohort of them would have reported a null on a contrast that could not have been non-null, spent
648 fits doing it, and read as evidence.

The two frozen trace fixtures already *are* the two regimes F4's row asks for — `v1`'s bound
slack on every row, `v2`'s clipping 54 of 200 — and they are committed files, so *declared in
advance* is literal rather than a promise about a seed. This is also
[`CLAUDE.md`](../../CLAUDE.md)'s own order of preference: an exact identity above a simulation,
and this is one of the places that rule pays.

## What the tests guard

- **the instrument does not move what it measures** — the reference arm is an ordinary `DRTMLE`,
  every patch restores every name it installed, and a raise inside a patched arm still restores.
  With a control that fails if the patch never ran, since three of those assertions would pass
  against a context manager that did nothing;
- **the R-style arm is R's round** — its route is `R_ROUND`, it is neither shipped order
  relabelled, the committed R record takes the same round, and F3's route and vintage gates pass
  for it. Instrument-validity only: nothing about `psi`, `se` or any curve;
- **the composition order that reads adoption** — `RStyleDRTMLE` before `TracingDRTMLE` in the
  MRO, pinned structurally because the failure is invisible in every number the fit reports;
- **each arm moves one factor** — `no-close` removes the closing pass and leaves the round count
  and exit reason where they were, and its final state is F2's recorded boundary to `1e-12`;
  `loose` is R's bar and moves neither the stall factor nor the unsolved bar; `raw` reaches
  `fit_reduced`'s bound and not the constructor's; `nested` moves one keyword;
- **the truncation reading is not vacuous** — `v1` is the inactive regime, `v2` clips more than a
  fifth of its rows, and something differs on `v2` at the initial fit. Without that last one the
  "inert" reading above is a test of nothing;
- **the run fails closed** — a moved rule, a changed configuration, a shared **data** seed between
  the cohorts, a draw outside the committed cohort, an incomplete draw set and an unknown cohort
  are each refused before anything is fitted;
- **the freeze is a freeze** — the cohorts are disjoint, reserving the sizing stream leaves them
  byte-identical, the smaller size takes a prefix of the larger, and the manifest round-trips.

## Four things this harness got wrong first, kept because they are the class of error F4 must not make

- **A contrast sized on the wrong variance.** The first design took E1b's `0.33` — a single arm's
  across-draw spread — for a study whose every outcome is a paired difference, and doubled it
  "for the unmeasured pairing gain". That is an invented multiplier in the one direction that
  costs fits, and it put the study at about six thousand of them: the size F4's row forbids by
  name. Sizing is now from a measured paired spread, per size, with the arithmetic in the
  artefact.
- **A stress design that could not be stressed.** A `bound-active` cohort was declared at
  `g_bounds = (0.15, 0.85)` before anyone measured whether that law's mechanism ever reaches a
  bound. It does not — clip share `0.0000` — so the design would have produced a confident null
  from 648 fits. **Declaring a regime is not entering one**, and the check is one fit.
- **A composition order that made the arm invisible.** `TracingRStyleDRTMLE` was first written
  with the tracing base first, which puts the partial adoption *inside* the recording closure: the
  recorder then stores the adopted set as what the refit produced, the next solve reads the same
  arrays, all three fields compare equal, and the route reads `refit:all`. The fit was right and
  the instrument was blind to it — the same class as F2's own recorded defects, *a right number
  filed under the wrong experiment*.
- **One factor with two implementations.** The raw-target convention existed as a subclass *and*
  as an inline closure inside the truncation reading. A mutation removing the factor from the
  subclass left the truncation tests green, because the reading never used it. Two copies of one
  factor, with the drift between them invisible — now one function, called from both, and the
  mutation fails.
