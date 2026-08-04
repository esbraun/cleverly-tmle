# DRTMLE: the investigation record

What was measured, what was hypothesised and dropped, and what one execution environment could
and could not reach. **This file is a record and not a plan** — [the roadmap](../roadmap.md) is the
plan, [the concordance](theorem-concordance.md) is what the sources say, and [the
validation plan](validation-plan.md) is what will be run. Everything here is kept for the
same reason `bench_tmle.py` keeps its own numbers: a measurement nobody can rerun becomes
folklore, and a hypothesis that was dropped without its reason written down gets proposed again.

It exists because the roadmap had become a status page, a methodology review, a forensic report,
an implementation spec, a test plan, a simulation protocol and a project history at once, and the
[second external review](#reviews-received) was right that the density made the dependency plan
harder to execute than the plan itself was.

## Reviews received

Two, both kept in full and neither adopted unread.

- **[The first review](review.md)**, verbatim, read the plan against Theorem 1 of
  Benkeser et al. (2017) and found the definition of done right and the route to it short by two
  conditions — now items 13 and 15. Three of its charges came back narrower than stated when
  checked against the code (its §1 terminology charge, §3's monotonicity charge, §7.1's on
  weights); the roadmap records each with what survives of it.
- **The second review**, of 2026-08-02, read the *revised* roadmap — the one that had turned the
  first review into a dependency-ordered execution plan — plus the 2016 Berkeley working-paper
  version of Benkeser, Carone, van der Laan & Gilbert. It accepted the item-20 diagnosis below,
  rejected three of the revision's claims, and supplied the theorem objects that
  [the concordance](theorem-concordance.md) is now seeded with. Its findings are
  distributed across the three live documents rather than kept as a document of their own,
  because unlike the first review it arrived with theorem text attached and that text belongs in
  the concordance rather than in a quotation of a review.

What the second review changed, in one place so the diff is legible:

| it said | the outcome |
| --- | --- |
| the item-20 root cause is right and the reordering follows | accepted; B1 is still first |
| B1 cannot both precede and depend on A1/A2 | accepted — B1 **splits** into B1a and B1b |
| R's clipping is not presumptively the right convention | accepted; four candidates, not two |
| "exactly two truncation conventions" is wrong | accepted and withdrawn |
| the weak-overlap fit must pass after B1 | accepted as an error; the criterion is now the identity, not the verdict |
| `α < 1/2` needs a nonzero-drift qualification | accepted; the drift coefficient is now designed and verified, not inferred |
| the mechanism correction's **sign** disagrees with the working paper | accepted as **item 21**, stop-ship — and **closed against the implementation's favour of it** once the paper itself was in hand: the §3.1 display it quoted is contradicted by that paper's own appendices ([lesson 10](#what-the-sizings-got-wrong)) |
| the theorem's update order is not the code's | accepted as **item 22**, and closed: the paper states its own exit as the three score equations, so the order is not prescriptive |
| statistical validity and product usefulness are two gates | accepted; the release rule splits |
| split the document by purpose | accepted; this file is part of that |
| soften "defect" on the multi-arm simplex question | accepted; it is an unasked question, not a known defect |
| move runner-specific network history out of the roadmap | accepted; it is [below](#what-one-runner-could-and-could-not-reach) |

## Item 20, from discovery to cause

### How it was found

Not by looking for it. Item 18 asked whether `repeats=` averages what it says it averages; the
cheapest way to check was a fit with two draws, and every test in `tests/unit/test_drtmle_fit.py`
until then had read **one fit on one split**. The second draw is the whole of why this was
visible: one draw clips no mechanism row and is centred, the other clips five and is not.

Over 24 draws — twelve `repeats=2` fits on `nonlinear_dgp` at `n=600`, `glm` on both nuisances,
`n_folds=5`, `learner_folds=3` — **six** leave `Pn[D*_Q + D*_g]` above `1e-8`, at magnitudes from
`2e-05` to `7e-04` on the scaled outcome. Every one of them exits on `"tolerance"`, with no
failure recorded and no ill-conditioned round. On one such draw equation (9)'s **recorded** score
is `3.7e-11` while the mean of the `D*_g` the curve actually subtracts is `−2.3e-04`.

It is not a `repeats=` defect and refusing that keyword would have misdiagnosed it: a draw of a
repeated fit is an ordinary fit, and the affected draws include first draws. It is not a
`nonlinear_dgp` defect either — that process is where it was seen because it is the module's
fixture, and a quarter of splits is the rate at which an ordinary `auto` bound binds on 600 rows.

`score_check` **does** catch it, on the *influence-curve* rows, which are computed from the curve
rather than from what the solver recorded. That is [item 16](../roadmap.md#closed-since-this-list-opened)
arriving on the first case nobody constructed for it, and it is the only reason this was seen at
all.

### The hypothesis that was checked and dropped

Both this page's first reading and an external execution plan read item 20 as a stale-array
defect: *"the recorded score and the reported curve are measured at arrays that are not the
same."* It is a reasonable reading, it put the fix behind a cross-language fixture, and it is
**false**.

Recomputing equation (9)'s score from the returned `fluctuation.mechanism.propensity` and
`fluctuation.reduction.reduced` reproduces the recorded score **bit for bit**, on an uncentred
draw and on a centred one alike. The record is faithful. The check that distinguished the two
hypotheses was thirty lines, one fit and no R, and it is the check that should have come first —
see [lesson 8](#what-the-sizings-got-wrong).

An immutable `DRTMLEState` object, which the execution plan proposed as the remedy, would not
have fixed this. Both expressions already read one state. What is not in that state is the
*truncation*, which two callers apply differently.

### The cause, algebraically

```text
equation (9), as solved     Pn[ H_g · (A − g*) ]        g* RAW,       from solve_mechanism
D*_g, as reported           Qr/ḡ* · (1_a − ḡ*)          ḡ* TRUNCATED, from reduced_corrections
```

Write `g_b = clip(g_raw)` and `H_g = Q_r/g_b`. The mechanism routine records

```text
S_raw = Pn[ H_g · (A − g_raw) ]
```

while the reported correction carries `D_{g,b} = (Q_r/g_b)·(1_a − g_b)`. Their difference is
exactly

```text
Pn[D_{g,b}] − S_raw = Pn[ (Q_r/g_b) · (g_raw − g_b) ]
```

so the two are **identical on every row the truncation does not bind and differ on every row it
does**; the magnitude depends on `Q_r/g_b` as well as on how many rows clip; and solving
`S_raw = 0` gives no guarantee whatever that `Pn[D_{g,b}] = 0`. For the lower arm the same
identity holds once the sign by which its residual is expressed through the binary upper-arm
mechanism is accounted for.

### The measurements

At `g_bounds="auto"` resolving to `[0.03191, 0.9681]` at `n = 600`:

| fit | rows clipped | recorded eq (9) | eq (9) at the truncated residual | reported `Pn[D]` |
| --- | --- | --- | --- | --- |
| `nonlinear` seed 3, draw 0 | 0 / 600 | `3.6e-11` | `3.6e-11` | `1e-09` — passes |
| `nonlinear` seed 3, draw 1 | 5 / 600 | `3.7e-11` | `−2.25e-04` | `ey0 +1.71e-03` |
| `nonlinear` seed 2 | 1 / 600 | `8.1e-11` | `4.35e-05` | `ey0 +5.82e-04` |
| `weak_overlap` seed 0 | 167 / 600 | `6.9e-11` | `−2.14e-04` | `ey1 +3.67e-03` |

The account is **quantitative, not merely directional**: the reported curve's mean is
`−range × Pn[H_g(A − ḡ*)]` to three significant figures on every one of them. On the
weak-overlap fit, `2.1354e-04` and `1.7037e-04` against an outcome range of `17.2` give
`ey1 = 3.673e-03`, `ey0 = 2.930e-03` and `ate = 7.43e-04`, which is what the score check reports
to the digit. Six seeded `nonlinear` fits give the equivalence in the other direction: **five with
zero clipped rows pass at `1e-09` to `1e-12`, and the one with a single clipped row of 600 fails
at `5.8e-04`.** One row is enough.

### A two-row construction that needs no simulation

The second review supplies this and it is worth keeping, because it proves the logical point
independently of anything measured here. Two rows, upper arm:

| row | `A` | `g_raw` | `g_b` | `Q_r` | raw score term | bounded correction term |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0.001 | 0.05 | 1.00 | −0.02 | −1.00 |
| 2 | 1 | 0.500 | 0.50 | 0.02 | 0.02 | 0.02 |

The raw score mean is exactly zero; the bounded correction mean is **−0.49**. Deliberately
extreme, and that is the point: no amount of solving the first equation bounds the second.

### What it settled

- **Items 11 and 20 are one defect.** The weak-overlap failure is not a structural
  incompatibility between aggressive truncation and the equations, not the conditioning of item 4,
  and not the exit criterion; it is this, at 167 clipped rows instead of one. The five places the
  first review widened the diagnosis to are all still worth *measuring* in B2, and none of them is
  the cause.
- **It is `DRTMLE`-only**, because it needs a `g` in a *residual*. Equation (8)'s covariate divides
  by `ḡ` and its residual is `Y − Q̄*`, so no plain `TMLE` fit can be in this state.
- **The `1e-11` is real and is measuring the wrong thing.** The loop solved the equation it posed.
  What it did not do is pose the equation whose solution the curve needs.
- **It is not [limitation 5](../roadmap.md#limitations-recorded-rather-than-fixed).** Those two
  looked like one story — "four to five orders worse on a quarter of draws" — and are not. The
  `1e-9` of limitation 5 is the equation the loop poses, measured at the arrays the loop leaves,
  and it stays `1e-9` on the uncentred draws too.
- **It is not [limitation 6](../roadmap.md#limitations-recorded-rather-than-fixed) either.** The
  closing pass's mechanism stage does bind on its cap, and the uncentred draws are the ones where
  the tilted `g*` leaves the bounds — but the cap binds on 94 of 96 fits while the curve is
  uncentred on a quarter of them, so the cap cannot be what selects them.

### What B1a measured when it landed

The patch is described in [the roadmap](../roadmap.md#what-b1a-landed); what belongs here are the
numbers it produced, because they are what the diagnosis above is now checked against on every
fit rather than once by hand.

On the module fixture — `nonlinear_dgp`, `n = 600`, seed 3, `glm` on both nuisances, `n_folds=5`,
`learner_folds=3`, reported on the **outcome** scale rather than the scaled one, so a factor of
`range = 15.13` against the figures in [the measurements](#the-measurements):

| draw | rows clipped | arm | `Δ_g` | `B_clip` | `Δ_Q` |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 / 600 | 0 | `5.5e-19` | `0` | `−1.1e-18` |
| 0 | 0 / 600 | 1 | `1.4e-18` | `0` | `0` |
| 1 | 5 / 600 | 0 | `3.410e-03` | `−3.410e-03` | `3.3e-19` |
| 1 | 5 / 600 | 1 | `−2.449e-04` | `2.449e-04` | `−9.0e-19` |

Three things in that table are the point. `Δ_g = −B_clip` **to every digit**, per arm, which is
the diagnosis reproducing itself as a number. `Δ_Q` is roundoff on every row, which is the control:
nothing truncates on equation (10)'s side, so an instrument that fired there would be broken rather
than informative. And the two draws differ, so the same fit carries both the case that binds and
the case that does not — a fixture with only one of them is [lesson
2](#what-the-sizings-got-wrong) waiting to happen.

The gap either side of the bar is what makes `IDENTITY_TOLERANCE = 1e-12` a threshold rather than
a guess: seven orders above the arithmetic, four below the smallest real failure seen (`7e-08`, on
an unseeded draw at `n = 400`).

**A clipping row does not by itself produce a large residual**, which is worth recording against
the obvious reading of item 20. `Δ_g` is `Q_r`-weighted at the clipped rows, so draw 1's two arms
differ by a factor of fourteen on the same five rows, and an unseeded `n = 400` draw with two
clipped rows came back at `6.9e-08`. "The bound binds" is the precondition; the magnitude is a
separate question, and a test that asserted the first while meaning the second would pass on draws
it should fail.

### What B1a found that was not item 20

The instrument's first run against a `guard=("g",)` fit reported a correction of `2.8e-03` on the
outcome scale at arm 1 with **zero clipped rows** and no equation (9) anywhere in the fit — which
is [item 23](../roadmap.md#what-is-still-open), a single-guard fit subtracting a correction it never
solved for. It is a different defect from item 20 in cause, in magnitude and in which fits it
touches, and nothing here would have seen it: no test in this repository fits a partial guard end
to end, and the default `guard=("Q", "g")` cannot be in this state.

That is [lesson 8](#what-the-sizings-got-wrong) arriving from the other side. Item 20 was found by
recomputing a recorded number from the returned state on one fit; this was found by the same
recomputation, once it was a permanent fixture of every doubly-robust fit rather than thirty lines
in a scratch file.

**Item 23 is now [closed](../roadmap.md#closed-since-this-list-opened)**, and the measurement above is
kept because it is the evidence rather than a symptom. The curve subtracts one correction per
equation the guard asked for; the unsolved equation's mean is still reported, as a diagnostic held
to no threshold, which is the row that found this. On the 600-row draw
`tests/unit/test_drtmle_fit.py` fits everything else on, the two arms read `1.2e-03` and `3.1e-04`
against a `5.4e-06` bar — the same finding on a second draw, and now the fixture of the first
partial-guard fit this repository has ever run end to end.

### What the B1b prototype measured

Sizing for [piece B1b](../roadmap.md#b1b--the-theorem-conforming-targeting-decision), which has since
landed on exactly the candidate this run selected. A scratch prototype rather than a deliverable:
two hooks on `estimators/targeting.py`'s module
namespace — the covariate builder, to capture the bounds, and `solve_mechanism`, to substitute a
candidate — so that a convention can be *fitted* rather than argued about. Nothing in the library
moved. The candidates are the validation plan's, and the first thing the run settled is that its
table's axis is not the discriminating one.

**The discriminating axis is which array the alternation carries forward, not which residual an
expression reads.** `solve_with_reduction` sets `targeted_g = mechanism.propensity`, the *raw*
tilted array, and the next round takes `logit` of that as its offset — so a row outside the bounds
stays outside them for the rest of the fit, and the disagreement with the curve's `ḡ*` is carried
rather than created afresh each round. Any convention that carries the **bounded** array forward
makes the identity hold at the exit near-automatically: the next round's offset is `logit ḡ*`, and
at a fixed point `ε → 0`, so the raw and bounded arrays coincide there. Both candidates measured
below exit with **zero rows clipped**, on draws where today's convention clips 1 and 167.

At `g_bounds="auto"`, `glm` on both nuisances, `n_folds=5`, `learner_folds=3`, `n = 600`, the
worst arm's `Δ_g` on the outcome scale and the worst arm's final equation-(9) score:

| draw | convention | rows clipped at exit | worst \|Δ_g\| | worst final score | `ate` |
| --- | --- | ---: | ---: | ---: | ---: |
| `nonlinear` seed 3 | current | 0 | `7e-19` | `1.0e-09` | `1.42968582` |
| `nonlinear` seed 3 | **D** | 0 | `4e-18` | `1.0e-09` | `1.42968582` |
| `nonlinear` seed 2 | current | 1 | `5.8e-04` | `1.1e-09` | `1.46604459` |
| `nonlinear` seed 2 | A | 0 | `7e-18` | `8.6e-10` | `1.46579069` |
| `nonlinear` seed 2 | **D** | 0 | `9e-19` | `7.8e-10` | `1.46328775` |
| `weak_overlap` seed 0 | current | 167 | `3.7e-03` | `1.2e-09` | `0.72277403` |
| `weak_overlap` seed 0 | A | 0 | `2e-18` | `5.6e-07` | `0.72664977` |
| `weak_overlap` seed 0 | **D** | 0 | `9e-18` | `6.6e-10` | `0.71406807` |

Three readings, and the third is the one that decides the piece:

- **The prototype reproduces the defect it was built to remove**, which is what says the hooks did
  not change the fit into something else: seed 2's `Δ_g` of `5.817e-04` and the weak-overlap
  draw's `2.930e-03` / `3.673e-03` are [the measurements above](#the-measurements) to the digit.
- **Where nothing clips, D is a regression surface.** On the module fixture — `nonlinear` seed 3,
  the draw `tests/unit/test_drtmle_fit.py` fits everything on — `psi`, `se` and both stored scores
  agree with today's fit to every digit printed. That is expected rather than lucky: with the clip
  slack on every row, D's equation *is* the logistic score, so the two solvers are chasing one
  root.
- **A and D separate only where the bound binds at the fixed point, and there they separate by
  four orders.** At the `auto` bound they are one fit. Forcing `g_bounds=(0.15, 0.85)` on the
  weak-overlap draw — 375 rows clipped under today's convention — leaves A's final scores at
  `6.8e-06` and `2.1e-06` against an inferential threshold of about `4e-06`, and D's at `2.1e-10`
  and `8.0e-10`. This is the predicted separation and not a numerical accident: A's substep solves
  the **pre-clip** logistic score and the clip is a projection applied after it, so a fixed point
  with clipping is a fixed point of neither equation, while D solves the equation the reported
  curve carries. `psi` moves from `0.7037` (A) to `0.7307` (D) against `se ≈ 0.08`, which is a
  third of a standard error and is the movement the validation plan says to investigate rather
  than to reject a candidate for.

**And it moves what B2 should expect.** On the `weak_overlap` seed-0 draw the fit as it stands
fails its score check with a verdict saying the standard errors do not describe the estimate;
under both A and D the same draw comes back with the identity at roundoff and its worst final
score at `5.6e-07` (A) and `6.6e-10` (D) against a threshold near `6e-06` — passing, both of them.
One draw is not the 24 that motivated a weak-overlap refusal, and B2 re-measures
all of them — but the standing instruction not to predeclare that refusal now has a measurement
behind it rather than only a caution.

**One thing it cost, and it is a condition to replace rather than a result.** B1a's fifth
condition is that an identity be checked *on a fixture where the bound binds*, witnessed by
`CorrectionRow.clipped`. Under any convention that carries the bounded array forward that witness
goes **vacuous** — `clipped` is 0 at the exit on every draw run above, including the one where 375
rows clipped before. A test selecting its fixture on `clipped > 0` would be selecting the empty
set, and one asserting `clipped > 0` would be asserting something that can no longer happen. That
is [stop-ship 14](../roadmap.md#stop-ship)'s shape — a check agreeing where it could not have
disagreed — arriving in a second place.

**And the replacement this run proposed was wrong, which the implementation found.** The clipped
share of the *initial* mechanism is a property of the draw, which is the right kind of thing, and
it is **zero on the draw item 20 was found on**: nothing about that fit's initial mechanism leaves
the bounds, and what clipped was the tilt. What works is `CorrectionRow.margin`, how close the
targeted mechanism comes to either bound as a fraction of the interval — `1.2e-06` on that draw
against `0.14` on its sibling, because a constrained root sits *against* the boundary of the
feasible set. It is not a proof that the constraint was active, and nothing derivable from the
returned arrays is, since the trajectory is not on the record.

**What the implementation then measured**, on the four fixtures above, against a threshold of
`4e-06` to `6e-06`: every state identity at `1e-17` or better, every final correction score at
`1e-09` to `1e-10`, and all four passing their score check. `psi` moved by `0` on the no-clip
fixture, `0.003·se` on `nonlinear` seed 2, `0.06·se` on `weak_overlap` at the `auto` bound and
`0.69·se` at the forced one — larger than the prototype's own figures at the same fixtures, since
the library's root finder and the prototype's hand-rolled Newton land on different iterates of the
same equation, and that difference is itself why the library does not use a hand-rolled one.

## How the alternation exits

96 fits: four processes by two sizes by twelve seeds, `glm` on both nuisances, `n_folds=5`,
`learner_folds=3`, both the data seed and the fold seed varying. Dispatched as
`.github/workflows/drtmle-convergence.yml` from `benchmarks/bench_drtmle.py`, 2,588s of runner at
42.6s per fit, and **no fit raised**.

**These numbers measure the exit criterion [item 7](../roadmap.md#closed-since-this-list-opened)
replaced**, not the current one. That is deliberate and is the order the item required — the
failure had to be characterised before the threshold moved — and re-measuring under the current
rule is [piece B2](../roadmap.md#b2--the-sweep-on-the-corrected-implementation)'s. **It has been
re-measured**, on the same grid and the same seeds, and every number below moved: [what the B2b
dispatch measured](#what-the-b2b-dispatch-measured). Read this table as the *before*, and do not
quote a cell of it as a live rate.

| process | n | rounds med [range] | tol/stall/cap | ill>0 | closing capped | med eq10 at exit | med min `mean\|h\|` | med `\|score\|/(se/√n)` | check fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | 600 | 22 [9–50] | 0/11/1 | 5/12 | 12/12 | 2.4e-09 | 9.8e-04 | 1.7e-07 | 1/12 |
| linear | 1,200 | 18 [9–50] | 0/11/1 | 9/12 | 12/12 | 1.1e-08 | 8.0e-04 | 3.2e-07 | 0/12 |
| nonlinear | 600 | 19 [13–50] | 0/10/2 | 0/12 | 12/12 | 2.5e-09 | 9.4e-03 | 1.8e-07 | 0/12 |
| nonlinear | 1,200 | 20 [13–50] | 0/11/1 | 4/12 | 12/12 | 9.0e-10 | 2.7e-03 | 1.9e-07 | 0/12 |
| off-diagonal | 600 | 16 [8–44] | 0/12/0 | 3/12 | 12/12 | 4.1e-09 | 4.8e-03 | 2.8e-07 | 0/12 |
| off-diagonal | 1,200 | 24 [8–50] | 0/11/1 | 3/12 | 12/12 | 1.4e-08 | 1.5e-03 | 4.0e-07 | 0/12 |
| weak-overlap | 600 | 16 [6–50] | 1/10/1 | 0/12 | 11/12 | 2.1e-11 | 3.1e-02 | 1.1e+00 | 11/12 |
| weak-overlap | 1,200 | 12 [7–50] | 1/10/1 | 0/12 | 11/12 | 8.7e-12 | 2.3e-02 | 1.0e+00 | 12/12 |

Read down the `tol/stall/cap` column first, because it is the one that changed a claim: **2 fits
of 96 reached the tolerance, 86 stalled and 8 ran out of rounds.** Then `ill>0`, which rises with
`n` on `linear` (5/12 to 9/12) and is highest exactly where the mechanism is easiest to get right
— the prediction [item 4](../roadmap.md#limitations-recorded-rather-than-fixed) makes and had never
tested. Then `check fails`, which is flat zero everywhere except `weak-overlap` and the one
`linear` draw at `n = 600`.

**That last column has an explanation and it is item 20.** `check fails` is not measuring the
alternation at all: it is measuring how often the tilted `g*` left the truncation bounds, because
those are the fits whose reported curve is not the one the solver solved for. That is why the
column is 23 of 24 on `weak-overlap` — where 29% of units sit outside the bounds on a seed-0 draw
— and flat zero on three processes whose bound rarely binds, and it is why the one `linear` draw
that fails does not otherwise look different from the eleven that do not. It also explains the
column's *independence* from `tol/stall/cap` and from `ill>0`, which had looked like the odd thing
about it.

**The zeros are not evidence the other processes are safe**: `nonlinear` shows 0 of 12 here and
1 of 6 on the seeds item 20 measured, so this column is sampling a per-draw event at whatever rate
the bound binds, not a per-process property. B2 re-runs the whole table with a clipped-row share
beside it.

## What the B2b dispatch measured

The same grid as *How the alternation exits* above, on the same seeds — 96 fits, four processes by
two sizes by twelve seeds, `glm` on both nuisances, `n_folds=5`, `learner_folds=3`, both the data
seed and the fold seed varying — dispatched as `.github/workflows/drtmle-convergence.yml` from
`benchmarks/bench_drtmle.py` at commit `6624e69`, and **no fit raised**. [Run
30907478598](https://github.com/esbraun/cleverly-tmle/actions/runs/30907478598), **378s of runner at
5.7s per fit**.

**That cost is the first finding and it was not a question anyone asked.** The same 96 fits took
2,588s and 42.6s apiece before, so the sweep is **seven times cheaper** — which is not a faster
machine but the same machine doing a seventh of the work, and is the exit distribution below
restated in seconds. Every sizing on this page and in [the validation
plan](validation-plan.md#4-the-sweep-piece-b2) was written against 42.6s a fit and is now wrong by
that factor; the 180-minute workflow cap, which the order arm was split in two to stay under, has
about thirty times the headroom it was thought to have.

### The exit distribution, under the rule that is actually in force

| process | n | rounds med [range] | tol/stall/cap | ill>0 | closing capped | med eq10 at exit | med min `mean\|h\|` | med `\|score\|/(se/√n)` | check fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | 600 | 9 [3–23] | 11/1/0 | 3/12 | 12/12 | 7.3e-05 | 1.6e-03 | 1.3e-07 | 0/12 |
| linear | 1,200 | 6 [2–22] | 12/0/0 | 3/12 | 12/12 | 6.7e-05 | 7.8e-04 | 1.7e-07 | 0/12 |
| nonlinear | 600 | 8 [4–33] | 12/0/0 | 0/12 | 12/12 | 4.5e-05 | 9.4e-03 | 4.4e-08 | 0/12 |
| nonlinear | 1,200 | 6 [3–50] | 11/0/1 | 2/12 | 12/12 | 5.9e-05 | 2.7e-03 | 2.1e-07 | 0/12 |
| off-diagonal | 600 | 4 [2–17] | 12/0/0 | 2/12 | 12/12 | 5.7e-05 | 5.1e-03 | 1.2e-07 | 0/12 |
| off-diagonal | 1,200 | 9 [2–31] | 12/0/0 | 2/12 | 12/12 | 4.9e-05 | 1.4e-03 | 3.0e-07 | 0/12 |
| weak-overlap | 600 | 5 [2–10] | 8/4/0 | 0/12 | 12/12 | 4.2e-06 | 3.4e-02 | 2.1e-07 | 0/12 |
| weak-overlap | 1,200 | 5 [2–8] | 9/3/0 | 0/12 | 12/12 | 5.0e-06 | 2.0e-02 | 2.4e-07 | 0/12 |

**`tol/stall/cap` has inverted.** It was 2 / 86 / 8 and it is **87 / 8 / 1**. Converging is what
the loop now mostly does, stalling is the minority, and exactly one fit of 96 ran out of rounds.
The median round count fell with it, from 12–24 to 4–9, and that is where the seven-fold saving
comes from. Two things are worth keeping apart here, because the temptation is to read this as the
alternation having been fixed: **what changed is which ruler the exit test uses**, not the
iteration. [Item 7](../roadmap.md#closed-since-this-list-opened) let an equation stop on an
absolutely negligible score rather than on a ratio to a covariate that vanishes, and the run's own
trailer says how often that branch is what fires: **96 of 96**, against 68 of 96 under the old
criterion. So it is not a branch that rescues a few hard draws — it is the branch every fit now
exits on.

**`ill>0` fell and kept its shape**, which is the more interesting half. It was 5/12 and 9/12 on
`linear` against 0/12 on `nonlinear` at `n = 600`; it is now 3/12 and 3/12 against 0/12. The
*prediction* [item 4](../roadmap.md#limitations-recorded-rather-than-fixed) makes — that `g_{r,2}`
vanishes where the mechanism is right, so the easy process is the ill-conditioned one — survives at
a third of the rate, and the rise with `n` on `linear` does not: 3/12 at both sizes. A
near-singular round is now something a fit passes through rather than something it exits at.

**`check fails` is flat zero. Every cell, both sizes, `weak-overlap` included.** It was 23 of 24
there, and the median standardised worst score on those cells was `1.1e+00` and `1.0e+00` — a score
the size of its own standard error. It is now `2.1e-07` and `2.4e-07`, five orders down and
indistinguishable from the three processes whose bound never binds. That is
[B1b](../roadmap.md#b1b--the-theorem-conforming-targeting-decision) measured at scale rather than on
four fixtures, and it is what [the roadmap's product
decision](../roadmap.md#b2b--the-dispatch-and-what-it-decides) is taken on: no refusal under weak
overlap, because the evidence that motivated one no longer exists.

### Where weak overlap enters, now that it does not fail

| process | n | clip share | margin | min g | ess/n | q99 h(8) | q99 h(9) | q99 h(10) | q99 `\|Qr\|` | min gr1 | q99 `\|gr2\|` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | 600 | 0.000 | 1.7e-01 | 0.226 | 0.46 | 3.1 | 2.61e-02 | 1.49e-01 | 1.24e-02 | 0.406 | 7.75e-02 |
| linear | 1,200 | 0.000 | 2.0e-01 | 0.244 | 0.47 | 2.9 | 1.77e-02 | 5.58e-02 | 7.30e-03 | 0.426 | 2.83e-02 |
| nonlinear | 600 | 0.000 | 1.3e-01 | 0.156 | 0.41 | 3.7 | 5.22e-02 | 2.28e-01 | 3.72e-02 | 0.128 | 1.63e-01 |
| nonlinear | 1,200 | 0.000 | 1.1e-01 | 0.142 | 0.42 | 3.7 | 6.19e-02 | 1.56e-01 | 4.18e-02 | 0.117 | 8.69e-02 |
| off-diagonal | 600 | 0.000 | 1.3e-01 | 0.156 | 0.43 | 3.5 | 1.71e-02 | 1.66e-01 | 6.44e-03 | 0.215 | 1.03e-01 |
| off-diagonal | 1,200 | 0.000 | 1.2e-01 | 0.142 | 0.42 | 3.5 | 1.35e-02 | 8.59e-02 | 5.79e-03 | 0.202 | 5.64e-02 |
| weak-overlap | 600 | 0.338 | 0.0e+00 | 0.000 | 0.13 | 10.7 | 1.40e-01 | 8.55e-01 | 8.61e-03 | 0.000 | 4.12e-01 |
| weak-overlap | 1,200 | 0.288 | 0.0e+00 | 0.000 | 0.09 | 10.0 | 7.23e-02 | 2.49e+00 | 4.84e-03 | 0.000 | 3.36e-01 |

**The draws are as hard as they ever were, and that is the point of reading this table beside the
one above.** A third of `weak-overlap`'s `(row, arm)` pairs lie outside the truncation at the
initial mechanism, the smallest `g` rounds to zero, the per-arm effective sample size is 9–13% of
`n` against 41–47% elsewhere, equation (8)'s covariate reaches 10 at the 99th percentile against
about 3, and `g_{r,1}` — a denominator — also rounds to zero. `margin` is `0.0e+00` on both cells
and `1.1e-01` to `2.0e-01` on every other, which is [B1b's
witness](../roadmap.md#what-b1b-landed) doing exactly what it was chosen to do: a constrained root
sits *against* the boundary of the feasible set, so a draw whose tilt wanted to leave the bounds
comes back pressed to one. So the fits that used to fail their score checks still have every
property that was blamed for it. What they no longer have is the failure, which is what says the
failure was the convention mismatch and not the overlap.

**`q99 h(10)` at `2.49` is the one column that still separates `weak-overlap` from everything
else** — an order above the `1.5e-01` to `2.3e-01` the other processes report, and *rising* with
`n` where every other cell falls. `g_{r,2}/g_{r,1}` with a denominator whose minimum is zero is the
third of §4's five places, and it is the one this dispatch does not clear. It does not currently
cost a fit its score check; it is where to look first if one ever fails again.

### What the reported curve rests on

| process | n | worst identity | worst B_clip | med std score | max std score | top 1% | top 5% | top 10% | med hessian cond |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | 600 | 4.3e-17 | 0.0e+00 | 8.40e-09 | 4.66e-08 | 0.10 | 0.34 | 0.56 | 7.5e+04 |
| linear | 1,200 | 2.8e-18 | 0.0e+00 | 1.24e-08 | 3.79e-08 | 0.10 | 0.33 | 0.53 | 1.1e+05 |
| nonlinear | 600 | 9.2e-18 | 0.0e+00 | 2.41e-09 | 3.16e-08 | 0.08 | 0.27 | 0.44 | 2.7e+03 |
| nonlinear | 1,200 | 5.7e-18 | 0.0e+00 | 5.67e-09 | 1.91e-08 | 0.08 | 0.27 | 0.42 | 5.0e+04 |
| off-diagonal | 600 | 1.9e-18 | 0.0e+00 | 1.24e-08 | 7.50e-08 | 0.07 | 0.23 | 0.40 | 1.0e+04 |
| off-diagonal | 1,200 | 1.5e-18 | 0.0e+00 | 1.94e-08 | 1.03e-07 | 0.12 | 0.34 | 0.55 | 9.6e+04 |
| weak-overlap | 600 | 1.6e-17 | 0.0e+00 | 1.83e-08 | 1.62e-07 | 0.28 | 0.46 | 0.60 | 1.3e+03 |
| weak-overlap | 1,200 | 3.1e-17 | 0.0e+00 | 8.41e-09 | 2.79e-07 | 0.33 | 0.55 | 0.71 | 5.1e+03 |

**Zero identity failures over 96 fits**, worst `4.3e-17` against a bar of `1e-12`, and `B_clip`
identically zero in every cell — including the two `weak-overlap` cells where a third of rows clip
at the initial mechanism. Items 11 and 20 were closed on four fixtures; this is the same closure at
scale, and it is [stop-ship 3](../roadmap.md#stop-ship) reduced to a column a reader can check.

The `weak-overlap` cells were also run at `n = 2,400` by the order dispatch below, and the trend
continues without breaking: `clip share` `0.231`, `ess/n` `0.08`, `q99 h(8)` `9.0`, `min g` and
`min gr1` still zero, `check fails` still `0/12`, and `closing capped` **11 of 12** — the first
cell anywhere in which the closing pass's mechanism stage reached its tolerance on a fit rather
than its cap.

**The concentration columns are the caveat this dispatch adds rather than removes.** On
`weak-overlap` the largest 1% of rows carry 28% and 33% of the worst score's absolute mass, against
7–12% on the three easy processes, and the top 10% carry 60% and 71% against 40–56%. A score driven
to `2e-07` by a handful of large rows cancelling is a different object from one that is small
rowwise, and only the second is something an interval rests on comfortably. Nothing here is a
failure — the standardised scores are five orders below their thresholds — but *passing the score
check* and *the score being well spread* are two properties, this dispatch measures both, and only
the first is on the face of a fit.

### The two update orders, against the yardstick of a fold split

Item 22's numerical half, at the configuration [§4 froze for
it](validation-plan.md#the-update-order-rule-frozen-before-the-dispatch) — three sizes, twelve
seeds, the paper arm and the reseed control, paired on the draw. **Two dispatches rather than one,
one process each**, `nonlinear` as [run
30907485303](https://github.com/esbraun/cleverly-tmle/actions/runs/30907485303) (108 fits, 722s,
9.8s a fit) and `weak-overlap` as [run
30907995192](https://github.com/esbraun/cleverly-tmle/actions/runs/30907995192) (108 fits, 393s,
6.4s a fit). **The split was a precaution against a 180-minute cap on a cost model that turned out
to be wrong by a factor of seven**, and neither run came close to needing it; it is recorded
because the validation plan promised one dispatch, and it changes nothing, since every table is
keyed on `(process, n)` and every clause is stated per process.

| process | n | med route `\|Δψ\|/se` | med reseed `\|Δψ\|/se` | route > reseed | paper `se` ratio | paper check fails | paper worst identity | reseed check fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nonlinear | 600 | 1.54e-01 | 1.49e-01 | 6/12 | 0.9880 | 0/12 | 8.8e-18 | 0/12 |
| nonlinear | 1,200 | 4.56e-02 | 8.81e-02 | 3/12 | 0.9940 | 0/12 | 4.0e-18 | 0/12 |
| nonlinear | 2,400 | 1.97e-02 | 7.25e-02 | 2/12 | 0.9981 | 0/12 | 5.4e-18 | 0/12 |
| weak-overlap | 600 | 1.64e-01 | 5.91e-01 | 3/12 | 1.0113 | 0/12 | 2.7e-17 | **1/12** |
| weak-overlap | 1,200 | 3.44e-01 | 4.50e-01 | 6/12 | 0.9939 | 0/12 | 1.1e-16 | 0/12 |
| weak-overlap | 2,400 | 1.15e-01 | 4.75e-01 | 3/12 | 0.9930 | 0/12 | 5.0e-17 | 0/12 |

**The rule's four clauses, taken in order and before anything is concluded from them:**

1. *median `|Δψ|/se` decreases across the three sizes in both processes* — **holds on `nonlinear`**
   (`1.54e-01 → 4.56e-02 → 1.97e-02`, a factor of 7.8 over a fourfold `n`) and **fails on
   `weak-overlap`**, which rises at `n = 1,200` before falling and ends only 30% below where it
   started;
2. *the count of draws where the route difference exceeds the reseed difference is compatible with
   half the pairs at the largest size* — `weak-overlap` is 3 of 12, which a paired binomial puts
   at `p ≈ 0.15` two-sided and is compatible; `nonlinear` is **2 of 12**, `p ≈ 0.04`, which is not
   compatible with half and **misses on the favourable side** — the route moves `ψ` *less* than a
   different fold split of one route does;
3. *median `se` ratio inside `[0.95, 1.05]` at the largest size in both* — **holds**, `0.9981` and
   `0.9930`;
4. *no fit in either arm fails its score check or its state identity* — **holds for both routes**:
   0 of 12 in all six paper cells and all six base cells, with worst identities at `1.1e-16` and
   better. One *control* fit fails, at `weak-overlap`, `n = 600`, and the control is a third arm
   rather than one of the two routes.

**So the rule is not met, and neither is what would falsify it.** The falsifier named in §4 is a
route difference that does not shrink while the reseed difference does; on `weak-overlap` the
reseed difference does not shrink either (`5.91e-01 → 4.50e-01 → 4.75e-01`). Both arms are noisy
at that overlap and the yardstick is as noisy as the thing being measured, so clause 1's failure
there is an **evidence-quality** outcome and not a finding about the routes. Written the other way
round: on the process where twelve draws resolve anything, every clause points the same way and the
route difference shrinks faster than a refit's; on the process where they do not, nothing is
resolved in either direction.

**What §4 says to do about exactly this is raise `--seeds`**, and it is what was done — see the
next subsection. Both readings are kept. Replacing a twelve-draw reading that failed with a
larger one that passes, and reporting only the second, is the shape of the mistake the rule was
frozen to prevent; the twelve-draw table above is therefore not deleted and its verdict is not
amended.

**Two things worth carrying beyond the rule.** The `weak-overlap` `se` ratio *ranges* are wide
where the medians are not — `0.5449` to `2.1648` at `n = 600`, still `0.8048` to `1.1939` at
`n = 2,400` — so a single weak-overlap fit's standard error is a good deal less stable than a
median over twelve makes it look. And the one control failure is the useful negative result here:
`weak-overlap`'s score checks pass on 0 of 24 in the main sweep and on 0 of 36 base fits at three
sizes, but **a different fold split of one of those draws does fail**. That is a rate near 1 in 100
rather than the 23 in 24 it used to be, and it is the reason the policy paragraph below stops short
of saying such a fit cannot fail.

### The same rule at thirty-six draws, and why the two readings are not nested

§4 names raising `--seeds` as the way to sharpen clause 2 and, by extension, the median clause 1
reads; the dispatch above cost 393s and 722s against a 180-minute cap, so it was taken. Three
sizes, **36 seeds**, both arms, one dispatch per process.

**They are not the same twelve draws plus twenty-four more, and this is the finding to carry
first.** `bench_drtmle.py` draws its three seed streams as `[:s]`, `[s:2s]` and `[2s:]` of one
`SeedSequence`, so raising `s` leaves the *data* seeds' prefix alone and moves the fold and control
blocks wholesale. A 36-seed run shares its first twelve **datasets** with a 12-seed one and **not
one of their fold splits**. The script's own comment says `generate_state` is prefix-stable, which
is true of adding a third *stream* — the change it was written for — and silent about raising
`--seeds`; it now says so. So the two readings are separate samples that happen to share some
draws, neither supersedes the other, and the honest statement is that the twelve-draw reading was
*underpowered*, not *wrong*.

| process | n | med route `\|Δψ\|/se` | med reseed `\|Δψ\|/se` | route > reseed | paper `se` ratio | paper check fails | paper worst identity | reseed check fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weak-overlap | 600 | 4.97e-01 | 6.87e-01 | 12/36 | 0.9938 | 0/36 | 3.0e-17 | 0/36 |
| weak-overlap | 1,200 | 2.89e-01 | 6.84e-01 | 6/36 | 0.9774 | 0/36 | 6.0e-17 | 0/36 |
| weak-overlap | 2,400 | 2.43e-01 | 4.32e-01 | 12/36 | 1.0024 | 0/36 | 2.2e-17 | 0/36 |

**Clause 1 holds on `weak-overlap` at 36 draws**, `4.97e-01 → 2.89e-01 → 2.43e-01`, monotone where
twelve draws gave `1.64e-01 → 3.44e-01 → 1.15e-01`. **Clause 3 holds**, `1.0024`. **Clause 4
holds**, 0 of 36 in every cell of both arms with identities at `6.0e-17` and better — and the lone
control failure the twelve-draw run reported is *absent*, which is the non-nesting again: that
failure belonged to a fold split this run does not contain. **Clause 2 is 12 of 36 at the largest
size**, which a paired binomial puts at `p ≈ 0.05` two-sided and which therefore misses
compatibility with half — **on the favourable side, as it did on `nonlinear`**: the route moves `ψ`
*less* than a different fold split of one route does, at every size.

**The reseed arm shrinks too, and more slowly**, `6.87e-01 → 6.84e-01 → 4.32e-01` against the
route's `4.97e-01 → 2.89e-01 → 2.43e-01` — a factor of 1.6 against 2.0 over a fourfold `n`. That is
the shape §4 called the expected finding: *a route difference that is large but shrinks at the same
rate as the reseed's is the opposite of the falsifier*, and here it shrinks somewhat faster.

**So the verdict, stated once and not re-argued.** Clause 2 is not met on either process, at either
seed count, and it misses in the direction that supports the routes agreeing rather than the one
that would count against them. Clauses 1, 3 and 4 hold on both processes once `weak-overlap` has
enough draws to resolve a median. **Item 22's numerical half is therefore not closed under the rule
as written**, and what stands between it and closure is a clause whose failures all point the wrong
way for a route difference — which is a thing to say plainly rather than to legislate away after
the fact. The rule may not be changed now. If a future revision wants to change it *before* a
further dispatch, the change with a reason behind it is that clause 2 should be **one-sided**: the
alternative it was written against is "the route moves `ψ` further than a split does", and a count
far *below* half is evidence for the conclusion, not against it.

## What the B2a smoke runs measured

Four fits at `n = 400` on `nonlinear`, which is the most of this that belongs in a small container
([`CLAUDE.md`](../../CLAUDE.md)), plus the two module fixtures the tests fit. They are here because
three of them are what [B2b](../roadmap.md#b2b--the-dispatch-and-what-it-decides) will have to explain
if the sweep reproduces them, and because a smoke run whose numbers are thrown away has to be
rerun by the next person.

**The two update orders do not always agree as closely as "the same fixed point" suggests.**
Paired on the draw, the same data and the same fold seed:

| draw | `\|Δψ\|` in units of `se` | `se` ratio (paper / this) | both score checks | rounds |
| --- | --- | --- | --- | --- |
| `nonlinear`, n=600, module fixture | 9e-03 | 0.977 | pass | 8 → 22 |
| `nonlinear`, n=400, sweep seed | 0.22 | 0.980 | pass | 15 → 15 |
| `nonlinear`, n=400, seed 3 | 7e-04 | 1.0006 | pass | 4 → 11 |

Every one of those fits solves all three equations at its returned state — `1e-09` to `1e-10`,
identities at `1e-18` — so none is unconverged and none is a defect. **It is what the paper's step
7 does not say**: the exit condition constrains three empirical *means*, and two states satisfying
them can differ in both `ψ` and `σ²_n`.

**And the 0.22 had no yardstick, which is the first thing the remediation fixed.** A route
difference and a *split* difference are the same number until something says which is which, so the
sweep gained a `reseed` arm — the same estimator, the same data, one different fold seed, paired on
the draw exactly as the paper arm is. At smoke scale on `nonlinear` at `n = 400`, which is **two
draws and resolves nothing**, the medians are `9.97e-02` for the route against `8.23e-02` for the
reseed, and the route moved `psi` further in 1 of the 2 pairs. That is the null this arm was built
to state: on present evidence the two update orders move `ψ` about as much as refitting one of them
on a different split does, and the "0.22 of a standard error" above is a draw of that distribution
rather than a property of the route. The dispatch that would settle it is
[the rule in §4](validation-plan.md#the-update-order-rule-frozen-before-the-dispatch)'s, and
the rule was written before it ran. The variance difference has a visible mechanism — the
routes exit holding reductions of different vintages, measured at `sd(g_{r,2})` of `0.024` against
`0.031` on the 600-row draw — and the `ψ` difference is the ordinary `o_p(n^{-1/2})` gap between
two asymptotically linear estimators of one parameter. **So the column to read at scale is
`|Δψ|/se` by size**: if both routes are asymptotically linear with the same curve it must shrink
with `n`, which is a claim with a direction rather than a reassurance, and one draw at each of two
sizes cannot see it.

**Two arms behaved exactly as their controls required**, which is what says the plumbing is
measuring the arm rather than the noise. `--reduced-learner glm` — the same learner the fit already
uses — reproduces the base fit to `0.0000` of a standard error and an `se` ratio of `1.0000`
exactly. `--truncation 0.02` on a draw whose smallest `g` is `0.128` is likewise *exactly* inert,
and `--truncation 0.25` on the same draw moves `ψ` by `0.078·se` and `se` by 0.9%. A truncation arm
that moved something where the bound cannot bind would have been reporting refit noise.

**Cost, which is what decides the dispatch.** 71s a fit at `jobs=1` in this container against the
42.6s a runner measured, and the paper's order took 22 rounds against 8 on the draw both were run
on — so `--order paper` should be budgeted at rather more than a doubling, not at one.

## What the oracle reductions measured

The sweep's oracle arm could not be built as specified — a reduction conditions on *fitted*
objects, so no DGP supplies its truth — but on the exact law it can, and
`tests/unit/test_oracle_reductions.py` is that: the law's own conditional expectations injected
through `ReductionSpec.refit`, recomputed at the current targeted pair every round as the fitted
ones are, with both primary nuisances **wrong on purpose** (`WRONG_G`, `WRONG_Q`) so that nothing
here is taken at the value where `Q_r` and `g_{r,2}` vanish.

| fit | reductions | `ey1` | `ey0` | `ate` | worst score | identity |
| --- | --- | --- | --- | --- | --- | --- |
| oracle | the law's own | `0.66000000` | `0.38000000` | `0.28000000` | `2.1e-11` | `2e-16` |
| saturated | `CellMeans` | agrees to `1e-9` | agrees | agrees | `2.1e-11` | `2e-16` |
| glm | linear in the design | `0.678434` | `0.387663` | `0.290771` | `7.0e-18` | `1.4e-16` |
| — | *the law's truth* | `0.66` | `0.38` | `0.28` | | |

Three readings, and the third is the one that changes what the arm is for.

**With the reductions exactly right the estimator recovers the truth**, to `3.6e-08` — with *both*
primary nuisances wrong, where a plain TMLE has no guarantee at all because its remainder is a
product of two errors and neither is zero. Under `guard=("g",)`, where no mechanism equation is
solved, it is exact to `1e-12`; that pair of numbers locates the `3.6e-08` at
[limitation 5](../roadmap.md#limitations-recorded-rather-than-fixed) — equation (9) is never solved
exactly, because its covariate reads the mechanism it tilts — rather than at the oracle being
approximate or the law being realised imperfectly. This is `test_remainder_drtmle.py`'s expansion
arriving at the other end of the estimator: that module shows *on paper* that one guard removes the
whole first-order remainder at exact reductions, and this is the production alternation doing it.

**The saturated learner reproduces the oracle to `1e-14`, over a whole alternation**, which is a
stronger statement than `test_reduced_regressions.py`'s one-call comparison: each round's
reductions decide the next round's covariates, so agreement at the exit says the two fits took the
same trajectory. It is the control — without it the injection could be computing something else
smooth and nothing would say so.

**A wrong reduction costs 0.36 to 0.80 of a standard error of *bias*, and leaves every score
solved.** The `glm` fit's worst correction score is `7e-18` — better-behaved than the oracle's —
while its `ate` sits `0.36·se` from a truth the oracle hits. So the discrimination this arm was
wanted for runs the *opposite* way from how §4 framed it: a sweep fit whose scores fail is not a
fit whose reductions were noisy, because a bad reduction does not show up in the scores at all. It
damages the estimate silently, which is the one failure an interval cannot report.

**The workflow was dispatched with all three new inputs before B2a was called landed**, at
`--processes nonlinear --sizes 400 --seeds 1 --order paper --reduced-learner glm --truncation 0.25`
— four fits, 89s, green ([run
30801387115](https://github.com/esbraun/cleverly-tmle/actions/runs/30801387115)). Two things it
established beyond "the YAML parses". The runner reproduced this container's numbers to every digit
printed — `2.21e-01`, `0.9804`, `1.0000`, `0.9914` — so the arms are deterministic across machines
and a sweep's numbers are a property of the seeds rather than of where it ran. And the runner is
the *faster* box: 36.0s a fit against 46s here at the same `jobs=2`. The point of dispatching a
smoke run rather than trusting the file is that this workflow has a history of being written while
every install step in the repository was broken, and dispatching it is what found that out.

The `order_control` input was verified the same way once it existed ([run
30807033826](https://github.com/esbraun/cleverly-tmle/actions/runs/30807033826) — six fits, 146s,
green), and it reproduced this container's `9.97e-02`, `8.23e-02` and `1/2` exactly. So the control
arm is deterministic across machines too, which is what lets its numbers be read against a later
dispatch's rather than only against themselves.

## What one runner could and could not reach

Kept here rather than in the roadmap because it is a property of one execution environment on one
date, and a plan that inherits it as a permanent obstacle is wrong twice over — this page carried
a *paywall* as the reason for two revisions, and the paywall was never the reason either.

The article is an NIH author manuscript deposited in PubMed Central as **PMC5793673**, so the full
text — Theorem 1 and appendices A to C — is obtainable without a Biometrika subscription.
Measured on **2026-08-02** from inside the Claude Code cloud sandbox: `pmc.ncbi.nlm.nih.gov`,
`europepmc.org`, `eutils.ncbi.nlm.nih.gov`, `biostats.bepress.com` (the working-paper mirror, UCB
paper 356) and `arxiv.org` each returned **403 at the agent proxy's `CONNECT`** — a network-policy
denial, not a paywall — while `raw.githubusercontent.com` and `pypi.org` returned 200.

Two documents have since been supplied by hand and neither required that network:
The first is Benkeser & Hejazi (2023), *Doubly-Robust Inference in R using drtmle*,
Observational Studies 9(2):43–78, and the 2016 Berkeley working-paper version of Benkeser, Carone,
van der Laan & Gilbert was read by the second review and transcribed into
[the concordance](theorem-concordance.md). **Check the network again rather than inheriting
this measurement**, and prefer a checked-in copy to either.

## What the sizings got wrong

Twelve lessons, distilled from the per-item retrospectives that used to run to several hundred
lines. They are kept and the retrospectives are not, because the only thing a retrospective is
for is the next sizing — the full pre-work read of what `drtmle` would touch, the per-seam record
of what each cost, and the six landed refusals' own notes are in git history, last carried in full
at `da8cacf`.

**1. A refusal's stated reason is the first thing to check, and it is wrong about half the time.**
Three of the six lifts in [Refusals worth lifting](../roadmap.md#refusals-worth-lifting) found the
written reason false rather than merely stale. `shifts=` refused `delta=`, `intermediate=` and
`weights=` together on one reason that was wrong for all three — conditional probabilities of
binary events do not become densities because `A` is continuous, and a weight tilts the population
rather than entering the clever covariate. `LTMLE`'s weights refusal claimed they "put a further
per-unit factor in the clever covariate's denominator at every node", and they do not. The
omitted-variable bound's refusal claimed `cf_d` was a coefficient in a treatment equation, and it
is not. In each case a one-line reason had been written once and never re-derived, and the lift
was smaller than the refusal implied. **Re-derive the reason before sizing the work.**

**2. The exact-law instrument goes blind in a predictable place, and it will not announce
itself.** Under a law the sample realises exactly with a saturated learner, anything that vanishes
at the truth is invisible: `DRTMLE`'s reduced regressions are identically zero row by row, so the
Gateaux modules supply a *degeneracy check* and would pass against a wrong sign or an omitted
term. Fixtures make maps the identity in the same way — a binary outcome makes `OutcomeScaler` a
no-op, so a residual taken against the raw outcome rather than the scaled one passed all 25 tests
of a new module; an empty `time_varying[0]` makes `history_frame(1)` the same object, so a
baseline-frame pin was blind. And at `epsilon = 0` the reported curve reads the observed block and
the untargeted `Q̄`, so a mechanism evaluated at the wrong dose passed all 39 tests of the module
written to catch it. What does see these: the **remainder idiom** at nuisances that are wrong on
purpose, a **structural pin** on the covariate's blocks, an **affine relabelling** of a continuous
outcome, and a **plug-in with `epsilon != 0`**. Of seven deliberate mutations on one module the
single survivor was of exactly this kind; of seven on another, three passed on the first try.
**Write the mutation before the test, and watch it fail.**

**3. A threshold changed after seeing a failure needs the failure characterised first.** One
unseeded fold draw ran to the outer cap with two scores at `1e-5`, and the response drafted was to
blunt `score_check` a thousandfold for every doubly-robust fit. The argument was sound and the
change was wrong: six seeded fits passed at the ordinary tolerance, so the failure was a minority
draw rather than a property of the estimator. The characterisation eventually took 96 fits and not
six, and it found something none of the three items it was run for was about — item 11. Blunting
the diagnostic would have hidden it. What survived the reverted change was the defect it turned up
on the way: the reported score and the reported curve were being read off *different* refits of
the reduced regressions.

**4. A test written after a change and never watched to fail pins nothing.** The exit-criterion
change passed all 61 `drtmle` tests identically before and after, because every assertion in the
suite is about the reported fit and the closing pass makes that fit the same either way. No
assertion about a *result* could have caught it; the predicate had to be unit-tested directly. The
corollary is the harder half: when the suite passes a change unchanged, that is evidence the
change is unpinned, not evidence it is safe.

**5. What the sizings got wrong about size was small; what they got wrong about *shape* was
not.** `drtmle` was sized at five to seven commits in `src/` and came out at four, so the count was
close. The shape was wrong twice, and both are general. It treated the reduced regressions as the
hard part and the alternation as transcription, and the alternation is where every surprise was —
a stall rule that had to watch the objective as well as the score, three scores that go stale in
three different ways, and equation (10)'s conditioning. And two orderings that looked free were
not: the **serializer bump belongs with the array**, not with the estimator that reads it, because
`NuisanceEstimates` is reconstructed field by field and a field added without one reloads silently
as `None`; and **the estimator and the curve had to land together**, because landing them apart
would have left `DRTMLE` reporting a plain TMLE's interval under a doubly-robust name for the
length of a commit, which is the failure mode the whole variant is organised around.

**6. The claims that last longest are the ones no test can fail, and only a reader catches them.**
Lesson 4 is that an unwatched test pins nothing; this is its complement. Items 13 to 19 are each
a *sentence* — the theorem's other assumption, what the fold reuse establishes, what a reader is
shown when the score check fails, the corrected curve's relationship to the efficient one, what
`weights=` needs said about it, what `repeats=` averages, what the monotonicity buys — and the
61-test `drtmle` suite is green with every one of them standing as it stood. Checked against the
code, three of [the first review](review.md)'s charges came back narrower than stated (14,
17 and 19, each still real) and the rest came back whole. That ratio is about the same as lesson
1's on refusals, and for the same reason: a written justification is a claim with no instrument,
so it decays at the rate claims with no instrument decay. **The cheapest instrument for a prose
claim is a reader who has the source open**, and one pass of that over this page cost less than
any item on it. The second review is the same lesson arriving with the *source itself* attached,
and it found [item 21](../roadmap.md#a1a--the-theoretical-audit) — a sign — which no reader without
the theorem could have found and no test in this repository would ever have failed against.

**7. A test can pin the wrong half of what it is named for, and then it decays like a prose
claim.** Piece 0 turned up two of these and neither was on this list. `TestItSurvivesARoundTrip`
had asserted a `DRTMLE` fit's estimates and curves came back intact for as long as the variant
had existed, and they did — while the *diagnostic* did not: `Fluctuation.mechanism` and
`.reduction` were never serialised, so a reloaded fit's score check answered a strictly weaker
question under the same name, and could pass where the live check failed. Limitation 8 recorded
that as lost record-keeping for two versions, which is what a defect looks like when it has been
written down in the wrong register. And every test in `test_drtmle_fit.py` read one fit on one
split, so item 20 — a quarter of splits leaving the reported curve uncentred — could not be seen
from inside the module until a second split was added for an unrelated reason. Lesson 4 asks
whether a test fails when the code is wrong; the two here failed to ask *which* wrong.

There is a third, about the roadmap rather than about the tests, and it is the same mistake one
level up. Item 20 was first written into *limitations recorded rather than fixed* — a section
whose own preamble says none of its entries would change a coverage number. An uncentred influence
curve is a variance estimate for something the fit did not compute, so it can change one, which
makes item 20 a link-4 failure and a piece's business rather than a limitation's. The entry itself
was accurate and cross-referenced from six places; it was the *heading above it* that told a
reader to discount it. **Where a finding is filed is part of what it says**, and a section preamble
is a claim about every entry under it — so adding an entry is asserting that preamble again.

**8. Two numbers that should be equal and are not is not yet evidence of two states, and
assuming it is cost this plan two revisions of its work order.** Item 20 was read, first here and
then by an external execution plan, as a stale-array defect. It was a reasonable reading, it put
the fix behind a cross-language fixture, and it was false — see
[the hypothesis that was dropped](#the-hypothesis-that-was-checked-and-dropped). The cheap check
that distinguishes the two hypotheses is the same either way and should have come first:
*recompute the recorded number from the returned state in the same process.* If it disagrees, the
state is stale; if it agrees, the state is fine and the two expressions are different functions —
and only the second hypothesis survives a fixture in another language, since R would have been
asked about the same two functions and answered about neither.

The general form is worth keeping because this variant will hit it again: **a truncation, a
scaling, a mask or a weight applied by two callers of the same array is a divergence with no
second state to find**, and it is invisible to every diagnostic that reads one side. Three of the
four are already in this code — `bound` on the mechanism, which is item 20; `OutcomeScaler`
between the equation's scale and the report's; and `observed`, which `reduced_corrections` applies
to `D*_Q` and not to `D*_g` where R applies it to both. The third is *latent* rather than live,
because `DRTMLE` refuses `delta=` and so no fit it accepts has a missing outcome — which is
exactly how the second one will look on the day someone lifts that refusal. The instrument is not
a state fingerprint. It is an **identity test between the stored score and a recomputation of the
term the curve carries**, which is the check the execution plan asked for under a diagnosis that
was wrong, and which is right regardless of the diagnosis — and regardless of which convention
[B1b](../roadmap.md#b1b--the-theorem-conforming-targeting-decision) eventually adopts.

**10. A display is not a derivation, and the difference decided item 21.** The charge that the
mechanism correction's sign disagreed with the theorem came from a faithful transcription of the
working paper's §3.1 display, made before the document itself was in hand. With the paper open,
the display is contradicted by the paper's own appendices twenty pages later — each derives its
block in a form satisfiable only with the *positive* correction, which is what both
implementations compute — and the same paper prints the other correction twice with two signs. So
the source had to be checked against **itself** before the code was checked against the source.
The general form: when a source and an implementation disagree, the first question is whether the
source disagrees with itself, and a quotation of one equation is not an answer to it. This is
[lesson 6](#what-the-sizings-got-wrong) sharpened — a reader with the source open beats a
quotation of the source, and beats a transcription of it for exactly the same reason.

**9. A finding located in the code is not a finding adjudicated against the theorem, and the
second review is where that cost showed.** Item 20's cause was found by recomputation, which is
the right instrument and answered the question asked: *are these two numbers the same functional
of the same state?* They are not, and the fix follows. What that instrument cannot see is whether
**either** expression is the one the theorem names — which on the sign of the mechanism correction
(item 21) took the paper itself, and where the answer could have been *neither*, leaving a fit
that satisfies the identity perfectly and reports the wrong variance. It was not neither; that
was not knowable from here. Parity with a reference implementation has the same blind
spot in the same place, and by construction: R and Python descend from one source, so agreement
is evidence about the transcription and not about the derivation. **Two checks that cannot fail
against the same class of error are one check**, however different their machinery.

**11. Before building an oracle, check whether the quantity collapses onto one already here — at
the value the check has to be taken at.** [A1a](../roadmap.md#a1a--the-theoretical-audit) was sized
as a further `discrete_law*` module carrying the whole DRTMLE limit as an analytic functional:
initial nuisances, three reduced regressions, the alternation, differentiated by complex step.
That is a few hundred lines, it re-encodes the algorithm rather than the derivation, and it would
have had to reproduce a loop [limitation 4](../roadmap.md#limitations-recorded-rather-than-fixed)
says does not reliably converge.

None of it was needed. The check has to be taken **in the union model**, because that is where
Theorem 1 applies — and there, at saturated reductions, the corrected curve collapses onto the
ordinary efficient influence function: `1/g_1 − g_{r,2}/g_{r,1} = 1/g_0` on the mechanism side,
and on the outcome side the `Q̄*` in `D*` cancels against the one inside `Q_r = Q̄_0 − Q̄*`. The
right-hand side was therefore already in the repository — `tests/discrete_law.py`'s `eif`, written
years earlier for the plain estimator — and what had to be written was the left. The module is one
file with no functional in it, it runs in under a second, and it closes to `1e-15` from a real fit.

The sizing error was not effort, it was **reading the plan's instrument as the requirement**. "Pin
the decomposition the way the `test_influence_gateaux*` modules do" was taken to mean *build what
those modules build*; what it meant was *compare against a derivative*. The general form: when a
plan names a technique, re-derive what the technique is for before costing it — the constraint
that makes the work necessary (here, the union model) is often the one that makes it small.

Two riders, because the collapse is not free. It holds only at **saturated** reductions, so the
module is silent about the pooled cross-fitting construction by construction — [stop-ship
14](../roadmap.md#stop-ship) exists so that silence is not later read as agreement. And it made three
mutations invisible, each found by running it and watching it *pass*. That half of the record is
the one that is never kept: a suite documents what it caught, and what it cannot catch is what a
later reader mistakes for coverage.

**12. The closing pass is an anaesthetic, so a defect in how the loop *exits* has to be caught at
the loop.** `_close_at_frozen_reductions` re-solves all three equations at the reductions the
record carries, and it is the last thing that runs — so whatever state the alternation exits in,
the *reported* fit is the state that pass leaves. Every assertion about `psi`, `se`, the curve, the
score check and the correction identities is therefore downstream of an operation that repairs the
thing they would have detected.

This has now happened twice and the second time was found only because the mutation was run.
[Item 12](../roadmap.md#closed-since-this-list-opened) was the first: the exit criterion changed and
"the whole 61-test `drtmle` suite passed identically before and after, because every assertion in
it is about the *reported* fit". The second is [B2a](../roadmap.md#b2a--the-sweep-instrument)'s
stale-score restatement — deleting it let the paper-order loop exit on a score for a state two
later steps had moved, and **68 of the module's 69 tests still passed**.

The general form, and it is a rule about where to put an instrument rather than about this loop:
*when a stage downstream of a loop recomputes what the loop was supposed to establish, no test of
the output can test the loop.* The three responses, in the order to reach for them: **remove the
asymmetry** so there is nothing to get wrong — which is what B2a's remediation did, making the
restatement unconditional once it was measured to be a bit-for-bit no-op on the default path;
**pin the invariant the repair rests on**, one level down, where it is still visible
(`tests/unit/test_fluctuation_score.py`); and only then **pin the call site structurally**, which
is what `tests/unit/test_sequential_design.py` does for a case where both variants are consistent
and no derivation separates them.

The trap in the third is that a structural pin *reads* like the other two. It says the code is
shaped a particular way, not that the shape is right, and a reader who finds one where an
invariant would have fitted will believe more than was checked.
