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
| nonlinear | 600 | 3.98e-02 | 1.65e-01 | 11/36 | 0.9994 | 0/36 | 1.1e-17 | 0/36 |
| nonlinear | 1,200 | 2.65e-02 | 9.34e-02 | 10/36 | 0.9994 | 0/36 | 1.5e-17 | 0/36 |
| nonlinear | 2,400 | 2.70e-02 | 9.68e-02 | 7/36 | 0.9995 | 0/36 | 6.6e-18 | 0/36 |
| weak-overlap | 600 | 4.97e-01 | 6.87e-01 | 12/36 | 0.9938 | 0/36 | 3.0e-17 | 0/36 |
| weak-overlap | 1,200 | 2.89e-01 | 6.84e-01 | 6/36 | 0.9774 | 0/36 | 6.0e-17 | 0/36 |
| weak-overlap | 2,400 | 2.43e-01 | 4.32e-01 | 12/36 | 1.0024 | 0/36 | 2.2e-17 | 0/36 |

**Clause 1 now holds on `weak-overlap` and fails on `nonlinear`, which is the exact reverse of the
twelve-draw reading.** `weak-overlap` runs `4.97e-01 → 2.89e-01 → 2.43e-01`, monotone where twelve
draws gave a rise at `n = 1,200`. `nonlinear` runs `3.98e-02 → 2.65e-02 → 2.70e-02` and rises by
**2%** in the last step, where twelve draws gave a clean factor of 7.8. So neither seed count
satisfies clause 1 on both processes, and the two disagree about *which* process is the problem.

**The honest reading of that is that the clause is measuring the median's noise rather than the
routes.** A 2% wobble in a median of 36 is not a claim about a limit, and the reseed arm wobbles
with it in the same place — `1.65e-01 → 9.34e-02 → 9.68e-02`, also up in the last step. Both arms
flatten between `n = 1,200` and `n = 2,400` together, which is a property of the pair of fits at
those sizes and not of either route.

**What is stable across both seed counts and both processes is the thing the control arm was built
to show.** The route difference sits **3.5 to 4 times below** the fold-split difference at every
`nonlinear` cell (`3.98e-02` against `1.65e-01`, `2.65e-02` against `9.34e-02`, `2.70e-02` against
`9.68e-02`) and below it at every `weak-overlap` cell too. §4 called *"a route difference that is
large but shrinks at the same rate as the reseed's"* the expected finding and the opposite of the
falsifier; what these show is a route difference that is **smaller than** the reseed's at every
size measured.

**Clause 3 holds everywhere** — `0.9995` and `1.0024` at the largest size, and `nonlinear`'s `se`
ratio is within `0.0006` of one at all three. **Clause 4 holds everywhere**, 0 of 36 in every cell
of both arms with identities at `6.0e-17` and better; the lone control failure the twelve-draw run
reported is *absent*, which is the non-nesting again — it belonged to a fold split this run does
not contain. **Clause 2 is 7 of 36 and 12 of 36** at the largest size, both short of half rather
than over.

**So the verdict, stated once and not re-argued.** Clause 2 is not met on either process at either
seed count, and clause 1 is met on one process at each seed count — a different one each time.
Every miss is in the direction that supports the routes agreeing: the route moves `ψ` *less* than a
refit of one route does, at every one of the twelve cells measured. **Item 22's numerical half is
therefore not closed under the rule as written**, and what stands between it and closure is two
clauses whose failures all point the wrong way for a route difference.

That is a thing to say plainly rather than legislate away after the fact, and the rule may not be
changed now. Two changes with reasons behind them, for a future revision to make **before** a
further dispatch: clause 2 should be **one-sided**, since the alternative it was written against is
"the route moves `ψ` further than a split does" and a count far below half is evidence for the
conclusion rather than against it; and clause 1 should be stated **relative to the control** — the
ratio of the two medians, which is stable here — rather than on the route median alone, which at
these counts is not.

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

## What the A1b dispatch measured

The nested construction against the pooled one, paired on the draw, with the `reseed` arm as the
yardstick — [run
30925185344](https://github.com/esbraun/cleverly-tmle/actions/runs/30925185344), at
`--processes linear nonlinear --sizes 600 1200 2400 --seeds 12 --order-control --reduced-crossfit
nested`. 216 fits in 884s at `jobs=2`, 5.4s median. `weak-overlap` and `off-diagonal` are left out
deliberately: [§7's rule](validation-plan.md#the-cross-fitting-rule-frozen-before-the-dispatch)
refuses to read a weak-overlap difference as evidence about item 15 on its own, since two of the
three reductions condition on `ĝ` and a third of that process's `(row, arm)` pairs clip, and the
budget bought a third **size** instead — which is what clause 1 needs.

| process | `n` | med nested `\|Δψ\|/se` | med reseed `\|Δψ\|/se` | ratio | nested > reseed | med se ratio |
| --- | --- | --- | --- | --- | --- | --- |
| linear | 600 | `2.13e-02` | `9.82e-02` | 0.22 | 3/12 | `1.0022` |
| linear | 1,200 | `1.35e-02` | `9.16e-02` | 0.15 | 2/12 | `0.9988` |
| linear | 2,400 | `1.82e-02` | `4.87e-02` | 0.37 | 3/12 | `0.9997` |
| nonlinear | 600 | `1.44e-01` | `1.49e-01` | 0.97 | 5/12 | `1.0216` |
| nonlinear | 1,200 | `9.64e-02` | `8.81e-02` | 1.09 | 7/12 | `1.0092` |
| nonlinear | 2,400 | `6.40e-02` | `7.25e-02` | 0.88 | 5/12 | `1.0039` |

**Three of the rule's four clauses pass and the primary one does not, so the rule does not
resolve.** That verdict is stated first because it is the one a later reader will be tempted to
soften.

- **Clause 2 passes**: the median `se` ratio at the largest size is `0.9997` and `1.0039`, both
  well inside `[0.95, 1.05]`.
- **Clause 3 passes**: the count of draws where the construction moves `ψ` further than a redrawn
  split does is 3/12 and 5/12 at the largest size, both at or below half — and below half in five
  of the six cells.
- **Clause 4 passes**: `0/12` score-check failures in every cell of both arms, and every state
  identity at `1e-17` or better (worst `9.6e-18`). The nested construction does not break the loop:
  across the base arm's 72 fits, 69 reached the tolerance, 1 stalled and 2 ran out of rounds.
- **Clause 1 — the primary — passes on `nonlinear` and fails on `linear`.** `nonlinear` is monotone
  and shrinks by 2.25x across the three sizes, faster than its own control's 2.06x. `linear` goes
  `2.13e-02 → 1.35e-02 → 1.82e-02`: down, then up.

**And on `linear` the failure has the literal shape of the falsifier**, which has to be said
plainly: the reseed difference halves across the sizes (`9.82e-02 → 4.87e-02`, 2.02x) while the
nested difference is flat (1.17x). *A construction difference that does not shrink while the reseed
difference does* is what §7 wrote down as the outcome that would send `reduced_crossfit=` to a
different default.

**It is not being read that way, and the reason is a number rather than a preference.** On
`linear` the nested difference sits **3 to 7 times below its own control at every size** — `0.22`,
`0.15`, `0.37` of it. The falsifier was written for a construction difference that *persists* while
split noise dies away; here the construction difference is a fraction of split noise throughout,
and what fails to shrink is a quantity already below the floor the control establishes. A slope
fitted to three medians of twelve draws at that magnitude is not measuring the construction. The
`se ratio range` says the same thing from another direction: at `linear`, `n = 600` it is
`0.1434 - 1.0185` in the nested arm and `0.1452 - 1.0395` in the reseed arm, so *one draw* is
pathological in **both** arms and has nothing to do with the construction.

**What is stable across all six cells, and it is the finding:** the construction difference is at
or below what a redrawn split moves — at every cell by the count, and at five of six by the median
— and on the process where the differences are large enough to have a trend, the two shrink
together. `Δ_k` behaves like split noise rather than like a persistent bias, which is the shape the
stability condition needs.

**This is the second time a median-based clause at twelve draws has failed to carry a slope
claim**, and that is worth filing against the instrument rather than against the estimator. [The
update-order rule](#the-same-rule-at-thirty-six-draws-and-why-the-two-readings-are-not-nested) hit
the same wall at twelve draws and again at thirty-six. The restatement §4 had already flagged as
available *before* a further dispatch — state clause 1 on the **ratio** of the two medians rather
than the arm's alone — does not rescue it either: the ratios are `0.22, 0.15, 0.37` and
`0.97, 1.09, 0.88`, neither monotone. What a further dispatch would need is not more seeds at these
sizes but the instrument §7 recorded as *not built*: the paired `L₂` distance between the two arms'
reduced arrays, which measures `‖Δ_k‖` itself rather than its consequence on `ψ`. The consequence
is where the cancellation and the noise are; the assumption is not.

**So item 15's empirical half is supported and not resolved, and the wording matters.** The
entropy condition is settled by the learner and is an argument rather than a run. The stability
condition has evidence pointing one way in every cell and a primary clause that does not close.
[Gate 1](../roadmap.md#c-the-demonstration)'s construction clause is met in the sense that the
decision is *frozen* — C fits the pooled construction — and the thing that would reopen it is a
`‖Δ_k‖` measurement, not another dispatch of this one.

## What C1's witness measured on its first run

[Item 25's witness](../roadmap.md#the-supported-contract-and-item-25) landed with piece C1 —
`CorrectionCheck.contract` and the two columns it needed, the initial mechanism's clipped count and
`g_{r,1}`'s signed margin — and it found something on its first run that the tables above could not
have shown. Recorded here rather than in [the design note](coverage-study.md) because it is a
measurement; the design note carries the consequence.

**A sixth to a third of well-overlapped draws exit outside the contract, and the initial mechanism
has nothing to do with it.** Six draws per cell of the Tier-1 coverage harness, on
`linear_dgp` — the *easy* process, chosen for overlap precisely so the cells would be inside the
contract:

| cell | `n` | bound-active | worst clip share | min margin | min gr1 margin |
| --- | --- | --- | --- | --- | --- |
| q-drift | 600 | 1/6 | 0.0000 | **0.0e+00** | 0.216 |
| q-drift | 1,200 | 2/6 | 0.0000 | **0.0e+00** | 0.406 |
| g-drift | 600 | 2/6 | 0.0000 | **0.0e+00** | 0.263 |
| g-drift | 1,200 | 0/6 | 0.0000 | 1.5e-01 | 0.344 |

Read this against [*Where weak overlap enters*](#where-weak-overlap-enters-now-that-it-does-not-fail).
That table puts `margin` at `1.1e-01` to `2.0e-01` on `linear`, `nonlinear` and `off-diagonal` and
at `0.0e+00` on `weak-overlap`, which reads as a clean separation between an easy process and a hard
one — and it is a table of **medians over twelve draws**. A minority at exactly zero is invisible to
a median, and the minority is what item 25's condition is about, since one bound-active draw in a
cell makes that cell's coverage number evidence about two estimators.

**And the cause is not positivity.** Two draws of the `q-drift` cell at `n = 600`, same settings,
same law, one pinned and one interior:

| | pinned (data `368974633`, fold `403478673`) | interior (data `2002320325`, fold `4034082052`) |
| --- | --- | --- |
| initial `ĝ(1\|W)` | [0.3464, 0.8631] | [0.4195, 0.8688] |
| `g_bounds="auto"` | (0.03191, 0.96809) | same |
| targeted `g*` | **[0.031910, 0.968090]** — both bounds attained | [0.155000, 0.688179] |
| mechanism `epsilon` | **24.47** | **0** |
| q99 `\|Q_r/g*\|` | 0.00734 | 0.0346 |
| `margin` | 0.0 exactly | 0.131 |
| closing steps / capped | 21 / yes | 21 / yes |

Equation (9)'s clever covariate is `Q_r/g*` and `Q_r = Q̄₀ − Q̄*` **vanishes where the outcome
regression is right**, which in `q-drift` it asymptotically is. So the score
`Pₙ[H₉(1_a − g*)] = 0` is solved against a covariate whose 99th percentile is `7e-03`, and its root
is an `epsilon` of 24 on the logit scale — which drives rows to *both* truncation bounds on a draw
whose initial mechanism never leaves `[0.35, 0.86]`. The interior sibling needed `epsilon = 0`:
equation (9) was already solved, because `Q_r` was already near zero.

**This is the mirror of [item 4](../roadmap.md#limitations-recorded-rather-than-fixed) and the half
of it that had not been written down.** Item 4 says `g_{r,2}` vanishes where the mechanism is right,
so equation (10)'s covariate is worst conditioned on the fits anybody wants. The same sentence with
the nuisances swapped says `Q_r` vanishes where the outcome regression is right, so **equation (9)**
is worst conditioned on exactly the off-diagonal cell in which `Q̄` is the consistent nuisance — the
cell the variant exists for. Both equations degenerate at the truth; the tables above only ever
recorded one of them doing it.

**It is not a defect and the wording is deliberate.** Every fit in the table passes its score check
and every state identity holds at roundoff, which is what `contract` being a *scope label* rather
than a verdict is for: on this evidence a fit that is sound in every way the package can check is
routinely outside the theorem's scope, and folding the label into `passed` would report a sixth of
well-behaved draws as broken. Six draws per cell is a share and not a rate — the pilot turns it into
one — but the pinned draw is one fit exhibited with its arithmetic, which is the right instrument for
a mechanism rather than for a frequency.

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

## What C2 measured, and the two things it changed its mind about

Piece C2 built item 13's instrument — the evaluation companion, Tier 2, and the remainder columns
— and two of its design choices are recorded here rather than in the design note, because both are
*measurements that overturned a plan* rather than constants a reader needs to look up.

### The regressogram §5 names cannot carry a declared remainder rate

The validation plan's §5 asks Tier 2 for *"a series, spline or histogram regression with a
smoothing sequence chosen in advance"*, and a histogram was the obvious reading. It does not work,
and the reason is §5's **own** inner-product trap arriving through a door that section does not
look at.

A regressogram's bias is `truth − within-bin mean`, which **oscillates in sign inside every bin**.
So its `L₂` norm is `O(B⁻¹)` while its inner product with a *smooth* weight is `O(B⁻²)` — the
oscillation integrates away against anything that does not oscillate with it. The remainder is an
inner product. So matching a declared remainder rate of `n^(−α)` needs `B_n ≍ n^(α/2)`... no:
`B_n^(−2) = n^(−α)` needs `B_n ≍ n^(α/2)`, which at `α = 0.25` and the study's sizes is 5 to 6 bins
and moves by 20% across a fourfold range — so integer rounding, not the sequence, decides the rate.
Pushing `B_n` up to make the rounding negligible puts the fit the other side of the bias–variance
line: an additive regressogram with `4 × 18` parameters on the 480 training rows of an `n = 600`
fold has a variance of `0.39` against a bias of `0.08`, and the remainder it then produces is
sampling noise wearing a design's name.

What replaced it is an oversmoothed **additive local-constant kernel**, whose bias is
`h²[½∇²m + ∇m·∇log p]` — smooth, and single-signed against a monotone weight, so no cancellation is
available to it. §5's list is illustrative and what it asks for is a sequence chosen in advance; a
bandwidth is one.

**The additive part is a second measurement.** A four-dimensional *product* kernel wide enough to
be bias-dominated at `n = 600` smooths over essentially the whole covariate space: measured at an
`L₂` error of **1.81** against an outcome standard deviation of **1.75**, which is not a slow
learner but a broken one. One dimension at a time has variance `O(1/(nh))` and is bias-dominated at
a bandwidth that still resolves the function.

### The committed coefficient came out within a few percent on the first draw

The design commits `c_ATE` by quadrature — `0.389` for `q-drift` and `0.410` for `g-drift`, sized so
they match Tier 1's `0.40`. One draw at `n = 600` measured `n^α R₂` at **`0.407`** and **`0.370`**
against them. That is §5's *"verify empirically that `n^α R₂ → c`"* landing on the first attempt,
and it is worth recording because the coefficient is a **prediction** here rather than the identity
Tier 1 normalises its shape to produce.

### Tier 2 is the cheap tier, which is the opposite of what the page expected

This page and the roadmap both said Tier 2 *"will not be"* cheap, on the reasoning that its
nuisances are fitted and that is what the stale 43s-per-fit figure was measuring. Re-timed on a
four-core container:

| | measured |
| --- | --- |
| a Tier-2 `DRTMLE` fit, `q-drift` at `n = 600`, 2,000-row companion | **5.4s** |
| the same at `g-drift` | **7.4s** |
| an ordinary `DRTMLE` fit at `n = 1,200` (C1's re-timing) | 5.6s |
| the figure piece C was costed from | 43s |

The additive smoother is cheaper than the `SuperLearner` the 43s was measuring, so the study is
affordable at either tier. The lesson is the one C1 already recorded and this is a second instance
of: **a cost figure inherited across a change of estimator is not a cost figure.**

### What a single replicate's `R_remaining` is mostly made of

`P₀D̂` is a quadrature over the evaluation draw, so its error is `sd(D)/√m` and it lands **directly**
in that replicate's `R_remaining`. Measured: at `m = 1,500` the error is `0.026` against a remainder
of order `0.007`, so one draw's column is three parts noise to one part signal. At `m = 4,000` it is
`0.016`. This is why the harness draws an **independent** evaluation sample per replicate — so the
error averages down across them rather than biasing every row the same way — and why every entry in
the remainder table carries a Monte Carlo standard error. A reader who takes a single row's
`√n R_remaining` as a measurement of the remainder is reading the quadrature.

**Those two figures are the ones on record, and E1 measured the same quantity a different way and
got a larger answer.** `sd(D)/√m` above was evaluated at an **assumed** `sd(D) ≈ 1.0`. Differencing
the two rules' measured spreads puts the draw's per-replicate error at `m = 2,000` at `0.033`
(`n = 600`) and `0.041` (`n = 2,400`) — `1.4x` to `1.8x` the figures above, which implies an
`sd(D̂)` between `1.5` and `1.8`. The direction is what matters: the paragraph above
**understates** how much of a replicate's column is the instrument.
[The ladder](#what-the-e1-ladder-measured) is the measurement.

## What the E1 ladder measured

> **Two readings in this section are withdrawn, and the tables are kept so the withdrawal is
> auditable.** `var removed` is not the share of the column that was the evaluation draw, and the
> `delta` ladder does not bound the grid's error. The reasoning is under each table and the
> replacement is [E1b](../roadmap.md#what-e1b-measures). Two things here are *not* withdrawn: the
> measured spreads themselves, which are standard deviations of a recorded column and need no
> identification argument, and the branch finding, which is a comparison of two
> `branch_error` columns rather than a variance decomposition -- and which survives its own
> retraction below, because a *disagreement between two independent rules* needs no error bound.
>
> **These numbers also came from a sandbox whose per-rung JSONL was git-ignored and is gone**, so
> they cannot be regenerated from retained evidence. The code they were produced at, `8341b78`, is
> the final code state of that pull request — `4f28116` touched only `docs/` — so there is no drift
> between the tables and the merged tree; what is missing is the rows. E1b's numbers come from a
> dispatched run with its artefacts recorded in [the manifest](study-manifest.md), and that is the
> gap this note exists to name.

*Two sweeps of `benchmarks/drtmle_companion_grid.py` at `8341b78`, on the four-core container
`CLAUDE.md` describes rather than on a runner, which is what the sizes and draw counts are chosen
for. Tier 1, both cells, `n ∈ {600, 2400}`, 16 draws, ladder `512/1024/2048/4096` Sobol points
against the i.i.d. rule at `m = 2,000`; 320 rungs in 382s at `jobs=3`. Tier 2, both cells, the same
sizes, 8 draws, ladder `512/1024/2048`; 128 rungs in 264s.*

### The two spreads are far apart, and how far apart they are is not the same as what caused it

The column is `var removed`: one minus the ratio of two measured across-draw variances of
`√n R_rem`, the deterministic grid's over the i.i.d. rule's, on the same draws through the same
primary fits. Tier 1:

| cell | `n` | spread, draw at `m = 2,000` | spread, grid at 8,192 rows | `var removed` |
| --- | --- | --- | --- | --- |
| `q-drift` | 600 | `1.0147` | `0.6192` | `0.628` |
| `q-drift` | 2,400 | `2.0505` | `0.3320` | **`0.974`** |
| `g-drift` | 600 | `0.9329` | `0.5018` | `0.711` |
| `g-drift` | 2,400 | `2.0969` | `0.2425` | **`0.987`** |

**What that column was read as, and why the reading is withdrawn.** The sentence this section
carried was *"at `n = 2,400`, 97% to 99% of the across-draw variance of `√n R_rem` was the
evaluation draw"*, and `var removed` does not estimate that share. Write `R_grid = X + e_grid` and
`R_draw = X + e_draw`, with `X` the estimator's own contribution. The pairing is better than the
paragraph above claimed — the companion is inert to the fit, so both arms are read through the
**same fitted curve** and `Var(X)` cancels *exactly* rather than approximately — and
`E[e_draw | fit] = 0` makes `Cov(X, e_draw)` vanish, because the i.i.d. companion is drawn
independently of the fitting rows. But `e_grid` is a **deterministic function of the fitted
curve** at a fixed grid, so `Cov(X, e_grid)` need not vanish and does not cancel. What the ratio
estimates is

```text
1 − s²_grid/s²_draw  →  [Var(e_draw) − Var(e_grid) − 2·Cov(X, e_grid)] / Var(R_draw)
```

and the target is the first term over the denominator alone. The slack is bounded by `ρ² + 2ρ` with
`ρ = sd(e_grid)/sd(R_draw)` — small if `ρ` is small, and `ρ` was estimated only by the `delta`
ladder, which the next section says estimates nothing. **What stands without any of that argument
is the two spreads**: `2.05` against `0.33`, and `2.10` against `0.24`, at `n = 2,400`.

**One part of the arithmetic here is unaffected and is worth keeping.** The rule's error is
`√n · sd(D̂)/√m` with `m` fixed, so it scales like `√n` while the estimator's own second-order
spread does not — which is why whatever the share is, it is larger at `n = 2,400` than at `n = 600`.
That is a statement about direction and it survives the retraction.

**What that would have bought a 250-draw dispatch**, arithmetically and at Tier 1: a Monte Carlo
error on the mean of `2.05/√250 = 0.130` under the draw against `0.332/√250 = 0.021` under the
grid in `q-drift`, and `0.133` against `0.015` in `g-drift` — a **sixfold** and an **eightfold**
narrowing. That much is a ratio of two measured spreads and survives the retraction above; it says
the grid's column is sharper without saying what the sharpening removed. C3c is a Tier-2 dispatch
and reported `1.252 ± 0.139` at `n = 2,400`, so these are not its numbers; what transfers is the
order.

### The ladder flattens, and flattening is stability rather than accuracy

`delta` is the paired movement of the column between a rung and the next coarser one. Tier 1,
`q-drift`, and all four of Tier 1's cell-and-size pairs behave the same way:

| `n` | 512 pts | 1,024 pts | 2,048 pts | 4,096 pts |
| --- | --- | --- | --- | --- |
| 600 | — | `0.01334` | `0.00663` | `0.00727` |
| 2,400 | — | `0.01641` | `0.01157` | `0.00875` |

**The reading taken off this table is withdrawn.** It was *"the finest rung's own error is between
one and three percent of the column's across-draw standard deviation"*, and a successive difference
does not bound an error without monotonicity or a convergence result that applies. The Tier-2
integrand has neither — E1's own module docstring says so of `_smooth_one`'s kernel cutoff, a jump
of `3.4e-4`, and then draws the opposite conclusion two paragraphs later.

**Measured rather than argued**, on this ladder's own geometry — `d = 4` as `linear_dgp`, the same
four rungs, standard normals through a scrambled Sobol sequence — with a piecewise-smooth integrand
of the kind the kernel cutoff produces, against a reference of 16 independent scrambles at `2^16`:

| points | `delta` | actual error | across-scramble sd |
| --- | --- | --- | --- |
| 1,024 | `0.000736` | `0.000188` | `0.001684` |
| 2,048 | `0.000717` | `0.000529` | `0.001097` |
| 4,096 | **`0.000102`** | **`0.000427`** | `0.000865` |

The finest rung's `delta` is **four times smaller than the error it was read as bounding**, and it
fails in the other direction too: on a smooth integrand at 1,024 points it *overstates* by three
orders. It is a stability statistic. What estimates the error is the third column — independent
scrambles of the same rule, which assume no rate at all — and that is what
[E1b](../roadmap.md#what-e1b-measures) reports in its place.

**Tier 2's `g-drift` at `n = 2,400` is the one place the ladder does not visibly flatten** —
`0.302` then `0.154` — and at eight draws a mean of eight absolute paired differences is itself a
noisy statistic. Under the reading above that is neither reassuring nor alarming: it is a statistic
about the sequence's movement and not about its error.

**The trade this column was offered as settling is real and is settled the other way instead.** A
fixed grid's error is a *bias* — the same points at every replicate — so unlike the draw's it does
not average down over a study, and a dispatch that could not bound it would be trading a measured
noise for an unmeasured one. The answer is not a better bound on the bias; it is to **randomise the
scramble** so there is no bias to bound, which is E1b's one mechanism and costs nothing.

### Tier 2's spreads move the same way, and its ratios are precise in one cell and useless in three

Tier 1's nuisances are prescribed, so its numbers are about the instrument and not about a fit a
learner produced; Tier 2 is the tier every rate on this page is quoted from and is where the reading
has to hold. Eight draws rather than sixteen, because a Tier-2 fit at `n = 2,400` in `g-drift` is
`20s` here — so these are **four estimates of a variance ratio from eight draws each**, and a share
read off them carries an error nothing in the table states.

| cell | `n` | spread, draw | spread, grid at 4,096 rows | `var removed` |
| --- | --- | --- | --- | --- |
| `q-drift` | 600 | `0.6900` | `0.7136` | `−0.069` |
| `q-drift` | 2,400 | `1.6183` | `0.2611` | **`0.974`** |
| `g-drift` | 600 | `1.6062` | `1.4327` | `0.204` |
| `g-drift` | 2,400 | `2.7971` | `1.8787` | `0.549` |

**The cell the rate is read in agrees with Tier 1 and the record said so far too strongly.** The
sentence was *"Tier 2 reproduces it exactly — `0.974` against Tier 1's `0.974`"*, and that is a
coincidence at the third decimal rather than a reproduction. The precision here is heavily
heteroscedastic, which the record missed by quoting one cell's uncertainty for all four. A
conservative independent-`F` interval — conservative because the pairing above is positive and only
tightens it:

| reading | draws | 90% interval on `var removed` |
| --- | --- | --- |
| `0.974`, Tier 1 `q-drift` `n = 2,400` | 16 | `[0.938, 0.989]` |
| `0.974`, Tier 2 `q-drift` `n = 2,400` | 8 | `[0.902, 0.993]` |
| `0.549`, Tier 2 `g-drift` `n = 2,400` | 8 | `[−0.708, 0.881]` |
| `−0.069`, Tier 2 `q-drift` `n = 600` | 8 | `[−3.05, 0.72]` |

So *"a variance ratio at eight draws has a standard error of roughly `0.5`"* is true where the
ratio is near one and badly wrong where it is near zero — the record wrote it for the negative cell
and then read it as a caveat on every cell including the headline. The negative reading is still
*no detectable difference at this count*, and it is also the direction the arithmetic points: the
rule's error grows like `√n` at fixed `m`, so its share is smallest at the smallest size — and
Tier 2's own remainder spread is the larger of the two there (`1.43` against Tier 1's `0.50` in
`g-drift`), which shrinks the share again. **All four of these are intervals on a statistic that is
not the share anyway**, per the first retraction; they are here because "the draw count made it
uncertain" was offered as the explanation and is not the right one.

**A second finding, which was not what the sweep was run for.** The appendix branches are resolved
an order more stably by the deterministic rule — `branch move` of `0.007` against `0.062` in
`g-drift` at `n = 600` — and under the draw that movement *exceeds* `|R_g|`, so the branch had not
settled there and had under the grid. In both `g-drift` rows the two rules disagree about `R_g`'s **sign**
(`+0.091` against `−0.014`, and `+0.066` against `−0.030`). The binned limits put 2,000 rows into
576 cells, which is three per cell; four times the rows is what buys the sign. That is a direct
reading on C3c's `branches settled` falling to `192/250`, and it is the reason
[the specification](validation-plan.md#reporting-r_q-and-r_g-separately) now reports `branch_error`
rather than only recording it.

**What survives the `branch move` retraction here, and what does not.** The *sign disagreement* does:
it is two independent integration rules answering differently about one quantity, which is evidence
without any error bound at all, and it is the strongest thing in this paragraph. The comparison
`0.007` against `0.062` does not become an error comparison — it is two **movements**, so it says
the grid's limits settled sooner and not that they settled nearer. See
[§5](validation-plan.md#reporting-r_q-and-r_g-separately).

### What it cost, which is nearly nothing

| tier | `n` | rule | rows | secs/fit |
| --- | --- | --- | --- | --- |
| 1 | 600 | grid | 8,192 | `7.5` – `8.1` |
| 1 | 600 | draw | 2,000 | `7.0` – `7.9` |
| 1 | 2,400 | grid | 8,192 | `5.6` – `12.0` |
| 1 | 2,400 | draw | 2,000 | `5.5` – `11.9` |
| 2 | 600 | grid | 4,096 | `5.1` – `8.4` |
| 2 | 600 | draw | 2,000 | `4.9` – `7.7` |
| 2 | 2,400 | grid | 4,096 | `9.9` – `20.6` |
| 2 | 2,400 | draw | 2,000 | `8.9` – `19.8` |

**Two to four times the companion rows for two to six per cent of the fit**, because the companion
costs a prediction per fold per nuisance and no learner fit — which is what
[C2's re-timing](../roadmap.md#what-c2-landed) already said and is why the grid could be taken
several rungs finer if a cell ever needed it. Tier 2 is the expensive tier and even there the
kernel smoother's cost is dominated by its training rows rather than by the rows it predicts at.

### What this table refuses to say, including in the direction that would flatter it

**It does not read the rate, and that refusal has to hold both ways.** The Tier-2 `q-drift` rows
have `√n R_rem` at `+1.62` at `n = 600` and `+1.05` at `n = 2,400` — a decline of `0.57` where
C3c's was `0.175` — and at eight draws with spreads of `0.71` and `0.26` that is about two Monte
Carlo standard errors. It is written down here because it is in the table and a table published
with a number nobody comments on is a number a reader will quote; it is **not** a rate, it is not
two seed batches, it is not 250 draws, and no clause of any gate is read off it. E1's scope is the
precision of the instrument and E5 is where the rate is read.

**What can honestly be said about resolvability is arithmetic and not a result.** At 250 draws the
Monte Carlo error on this column would be several-fold smaller under the grid than under the draw:
sixfold and eightfold at Tier 1, and `0.261/√250 = 0.017` against `1.618/√250 = 0.102` in Tier 2's
`q-drift` at `n = 2,400`. Whether that separates a decline depends on the decline, which is the
thing not being measured here.

**The trap runs both ways, and this section was drafted into it once before it landed.** The
tempting sentence is *"and it still would not resolve"*, which the record's own
`sd(D)/√m ≈ 0.023` would have supported — and which [lesson
19](#what-the-sizings-got-wrong) finds was an assumed figure rather than a measured one. It is as
much a reading of the rate as the opposite, and the tables above happen to point the other way.
Neither belongs here.

**And what a quadrature's own error bounds is the quadrature.** No refinement or randomisation of
an integration rule can detect a defect in the estimator, and every column here integrates
`dgp.propensity` and `dgp.outcome_mean` against predictions of the same functions. That refusal was
right in E1 and is untouched by either retraction — what the section got wrong is how large the
instrument's error is, not what knowing it would license.

## What the E1b dispatch measured

*Two dispatches of `.github/workflows/drtmle-companion-grid.yml`, tier 1 and tier 2, at
`79d11d3252d784c2c0f93c67aa4f7e31630f22c6`. Both cells, `n ∈ {600, 2400}`, **32 draws** a cell and
size, **8 independent scrambles** of the quasi-random rule and **8 independent i.i.d. companions**
of 2,000 rows per fit, ladder `512/1024/2048` at tier 2 and `512/1024/2048/4096` at tier 1. One fit
a draw: 256 fits, and 1,024 or 1,280 replicate rows a job. Runs `31021187807` and `31021176323`;
eight artefacts, [manifested](study-manifest.md#e1b-what-was-run) with their digests. Unlike
[E1's](#what-the-e1-ladder-measured), these rows are retained.*

### Each rule's own error, measured rather than derived

`rule sd` is the standard deviation of `√n R_rem` across **that rule's independent replicates at a
fixed fit**. It needs no convergence rate, no halving witness and no comparison with the other rule.
`share` is `Var(e) / (Var(X) + Var(e))` with both terms estimated — the fraction of a
**one-replicate** study's across-draw variance that the rule accounts for — and `share 90%` is a
bootstrap over draws. At the finest rung of each ladder:

| tier | cell | `n` | `rule sd`, grid | `share`, grid | `rule sd`, draw | `share`, draw |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `q-drift` | 600 | `0.0029` | `0.000` `[0.00, 0.00]` | `1.1299` | **`0.821`** `[0.73, 0.93]` |
| 1 | `q-drift` | 2,400 | `0.0041` | `0.000` `[0.00, 0.00]` | `2.1656` | **`1.003`** `[0.95, 1.05]` |
| 1 | `g-drift` | 600 | `0.0142` | `0.001` `[0.00, 0.00]` | `1.1382` | **`0.808`** `[0.69, 0.95]` |
| 1 | `g-drift` | 2,400 | `0.0069` | `0.001` `[0.00, 0.00]` | `2.2033` | **`1.008`** `[0.96, 1.06]` |
| 2 | `q-drift` | 600 | `0.0326` | `0.002` `[0.00, 0.00]` | `1.2181` | **`0.723`** `[0.60, 0.89]` |
| 2 | `q-drift` | 2,400 | `0.1399` | `0.081` `[0.02, 0.11]` | `2.1839` | **`0.991`** `[0.94, 1.05]` |
| 2 | `g-drift` | 600 | `0.1804` | `0.019` `[0.01, 0.04]` | `1.5778` | **`0.640`** `[0.53, 0.80]` |
| 2 | `g-drift` | 2,400 | `0.4671` | `0.051` `[0.02, 0.10]` | `3.2591` | **`0.790`** `[0.69, 0.91]` |

**At `n = 2,400` the evaluation draw accounts for essentially all of the across-draw variance of
`√n R_rem` under C3c's rule** — `0.991 [0.94, 1.05]` in the tier-2 cell the rate is read in, and
`1.003` and `1.008` at tier 1 — and for two thirds to four fifths of it at `n = 600`. Every one of
those is a ratio of two estimated variances rather than a difference of two marginal ones, so
nothing in it rests on how the rule's error relates to the estimator's.

**E1's withdrawn column got the tier-1 headline about right and could not have known it.** Its
`var removed` read `0.974` and `0.987` where these read `1.003` and `1.008`; its `−0.069` and
`0.204` at `n = 600` are `0.723` and `0.808` here. The point of the retraction was never that the
numbers were far out — it was that `1 − s²/s²` does not estimate that share and that a ladder does
not bound the term it was corrected by, so two of the four cells came out uninformative and no cell
came with an interval.

**A share above one is a draw count and not a finding**, and three cells reach past it at the top of
their intervals. `Var(X)` is estimated as a *difference* — the between-fit variance less
`Var(e)/R` — and where the rule's error dominates by an order that difference is not resolvable at
32 draws: the two rules' `est sd` read `0.3216` and `0.0000` in tier 1 `q-drift` at `n = 2,400`.
Neither is clipped, because clipping would make an unresolved reading look resolved.

### The grid's error is two to three orders below the draw's, and now that is a measurement

At the finest rung, `rule sd` of `0.0029`–`0.0142` at tier 1 and `0.0326`–`0.4671` at tier 2,
against `1.13`–`3.26` for the i.i.d. rule at `m = 2,000`. Tier 2's is the larger and the reason is
its integrand: both nuisances are fitted there and the kernel smoother's cutoff makes the integrand
only piecewise smooth, which is exactly the case a quasi-random rule handles least well — and
exactly the case E1's ladder was least able to see.

**The ladder still flattens and it is still not a bound.** Tier 1 `q-drift` at `n = 2,400` reads
`delta` of `0.01391 / 0.00735 / 0.00386` down the rungs while `rule sd` reads
`0.0158 / 0.0076 / 0.0041` — close, because a smooth integrand is the case where a successive
difference happens to track the error. Tier 2 `g-drift` at `n = 2,400` reads `delta` of
`0.28625 / 0.20600` against `rule sd` of `0.6919 / 0.4671`: the movement is **less than half** the
error, in the cell whose integrand is the rough one. That is the retraction, reproduced on the
estimator's own integrand rather than on a constructed one.

### What it cost, which is more than E1 and still small

| tier | `n` | companion rows a fit | secs/fit |
| --- | --- | --- | --- |
| 1 | 600 | 81,536 | `13.8`–`16.0` |
| 1 | 2,400 | 81,536 | `7.3`–`21.4` |
| 2 | 600 | 48,768 | `21.5`–`22.5` |
| 2 | 2,400 | 48,768 | `60.5`–`60.6` |

Ten to twenty times E1's companion rows for two to three times its wall clock, because a companion
row is a prediction per fold per nuisance and no learner fit. The whole record above is 256 fits and
about 70 minutes of runner time across eight jobs.

### What this dispatch refuses to say

**It reads no rate.** `√n R_rem` moves between the sizes in these tables and none of that is a
finding: 32 draws is an instrument-sizing count, the cells are not two seed batches, and E5 is where
a rate is read against clauses frozen before it. **It selects no learner**, which is E2's and E2b's.
And a rule's own error, however well measured, says nothing whatever about the remainder — every
column here integrates `dgp.propensity` and `dgp.outcome_mean` against predictions of the same
functions.

**What it does license is a sizing.** The draw's share at `n = 2,400` means C3c's `± 0.09` was
almost entirely instrument, and that under the randomised rule the same replicate count buys a
Monte Carlo error several-fold smaller. Whether that separates a decline depends on the decline.

## What the E2 dispatch measured

*One dispatch of `.github/workflows/drtmle-reference.yml` at tier 2, at the workflow's own defaults
with **no inputs passed**, on `main` at `6f3aeb38ee1e23fc06ea598c6e511d2e686457bf`. Both cells,
`n ∈ {600, 2400}`, 32 draws a cell and size, reference `spline(16)` on 4,096 Sobol points against
`glm`, scored on 8,192, evaluated on 2 × 2,048, gate C at 4 scrambles on 4 draws. 76 fits a job,
304 in all; run `31042558057`, four artefacts,
[manifested](study-manifest.md#e2-what-was-run) with their digests.*

**The headline is that three cells of four cannot be read, and the reason is the falsifier
[§8](validation-plan.md#8-the-reference-comparison-piece-e2) wrote down before the run.** E2 was
built to branch — `moved` fires E2b, `equivalent` shuts the learner road — and it branches on
neither, because the reference whose gates fail is a reference no comparison answers for. **That is
the gate doing its job rather than the run failing**, and the distinction is the whole reason the
gates were specified before any paired number existed.

### The gates, which is the table to read first

`B. risk vs X` is candidate `X`'s held-out weighted risk **minus** the shipped reference's, on rows
neither saw: positive means the reference is the better estimate of that function. A **difference**
and never a ratio. The three reductions are read apart.

| cell | `n` | reduction | `bins(8)` | `spline(32)` | `spline(8)` |
| --- | --- | --- | --- | --- | --- |
| `q-drift` | 600 | `qr` | `+2.036e-06` | `+3.702e-07` | `+6.125e-08` `[-2.42e-08, +1.55e-07]` |
| `q-drift` | 600 | `gr1` | `+2.727e-05` | `+1.663e-05` | `-3.116e-06` `[-7.76e-06, +2.14e-06]` |
| `q-drift` | 600 | `gr2` | `+9.603e-04` | `+8.974e-05` | **`-2.855e-05`** `[-4.72e-05, -1.21e-05]` |
| `q-drift` | 2,400 | `qr` | `+8.086e-07` | `+6.428e-08` | `+9.462e-09` `[-2.27e-09, +2.33e-08]` |
| `q-drift` | 2,400 | `gr1` | `+1.057e-05` | `+2.812e-05` | **`-1.550e-05`** `[-1.84e-05, -1.29e-05]` |
| `q-drift` | 2,400 | `gr2` | `+6.362e-04` | `+5.065e-05` | **`-2.350e-05`** `[-2.79e-05, -1.94e-05]` |
| `g-drift` | 600 | `qr` | `+4.896e-05` | `+3.832e-06` | `-2.180e-07` `[-9.04e-07, +5.96e-07]` |
| `g-drift` | 600 | `gr1` | `+1.276e-04` | `-5.891e-06` | `+3.132e-05` `[+5.19e-06, +6.44e-05]` |
| `g-drift` | 600 | `gr2` | `+3.487e-04` | `-8.039e-05` | `+1.412e-04` `[-2.53e-05, +3.84e-04]` |
| `g-drift` | 2,400 | `qr` | `+5.384e-05` | `+4.551e-06` | **`-1.678e-06`** `[-2.05e-06, -1.35e-06]` |
| `g-drift` | 2,400 | `gr1` | `+1.287e-04` | `-1.074e-05` | `+2.744e-05` `[-5.07e-07, +6.95e-05]` |
| `g-drift` | 2,400 | `gr2` | **`-1.031e-04`** `[-4.35e-04, +1.12e-04]` | `+1.164e-04` | `-1.361e-04` `[-4.14e-04, +3.29e-06]` |

| cell | `n` | gate B | gate C | why |
| --- | --- | --- | --- | --- |
| `q-drift` | 600 | **fail** | `0.0665` against `0.1283` | `spline(8)` beats the reference on `gr2` |
| `q-drift` | 2,400 | **fail** | `0.0091` against `0.0966` | `spline(8)` beats it on `gr1` **and** on `gr2` |
| `g-drift` | 600 | pass | `0.1662` against `0.2999` | — |
| `g-drift` | 2,400 | **fail** | `0.1217` against `0.3401` | `spline(8)` beats it on `qr`; `bins(8)` **not rejected** on `gr2` |

**Gate C passes in all four**, by a factor of two to eight, which is the half the
[three-draw pilot](validation-plan.md#8-the-reference-comparison-piece-e2) sized `--reference-points`
against and the half that came out as designed. **Gate B is where it fails**, and every failure is
the same clause: *no other rung may be strictly better*.

**Which rung wins differs by cell and by reduction, and that is the diagnosis rather than a
detail.** §8 named it in advance — *"a gate-B ordering that disagrees between the reductions — the
shipped rung best on one and beaten on another — says the reference's resolution is not one choice,
and the repair is a per-regression resolution rather than a verdict."* That is exactly the pattern:
`spline(8)` beats `spline(16)` on `gr2` at `q-drift` 600, on `gr1` and `gr2` at `q-drift` 2,400, and
on `qr` at `g-drift` 2,400 — and on nothing at all at `g-drift` 600. **The coarser rung is the better
estimate of some of these functions and the worse estimate of others**, so one knot count for three
regressions is the thing that is wrong, not the ladder.

**`g-drift` at `n = 2,400` fails a second clause and it is the more serious one.** `bins(8)` — the
negative control the gate exists to reject — reads `-1.031e-04 [-4.35e-04, +1.12e-04]` on `gr2`,
which is not a rejection: at that size and on that regression the gate cannot discriminate a
deliberately coarse arm from the reference at all. A gate with no teeth is a gate that cannot pass
anything, which is why it reads `fail` rather than being ignored.

### The comparison, which three cells do not license reading

| cell | `n` | estimand | `glm` | reference | `paired d` | `d 95%` | margin | `rule se` | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `q-drift` | 600 | `ate` | `+1.5401` | `+0.1685` | `-1.3623` | `[-1.6564, -1.0821]` | `±0.3850` | `0.0094` | `unresolved` |
| `q-drift` | 600 | `ey1` | `+0.8081` | `+0.1436` | `-0.6785` | `[-0.9075, -0.4635]` | `±0.2020` | `0.0061` | `unresolved` |
| `q-drift` | 600 | `ey0` | `-0.7320` | `-0.0250` | `+0.6838` | `[+0.5508, +0.8190]` | `±0.1830` | `0.0043` | `unresolved` |
| `q-drift` | 2,400 | `ate` | `+1.1594` | `+0.1114` | `-1.0491` | `[-1.1667, -0.9096]` | `±0.2899` | `0.0185` | `unresolved` |
| `q-drift` | 2,400 | `ey1` | `+0.5568` | `+0.0512` | `-0.4985` | `[-0.6046, -0.3567]` | `±0.1392` | `0.0180` | `unresolved` |
| `q-drift` | 2,400 | `ey0` | `-0.6027` | `-0.0602` | `+0.5507` | `[+0.4885, +0.6131]` | `±0.1507` | `0.0024` | `unresolved` |
| `g-drift` | 600 | `ate` | `+3.5989` | `+1.9730` | `-1.7893` | `[-2.2739, -1.3569]` | `±0.8997` | `0.0417` | **`moved`** |
| `g-drift` | 600 | `ey1` | `+1.6345` | `+0.9383` | `-0.7677` | `[-0.9343, -0.6032]` | `±0.4086` | `0.0354` | **`moved`** |
| `g-drift` | 600 | `ey0` | `-1.9644` | `-1.0348` | `+1.0216` | `[+0.6675, +1.4366]` | `±0.4911` | `0.0147` | **`moved`** |
| `g-drift` | 2,400 | `ate` | `+4.0814` | `+2.4033` | `-1.6570` | `[-2.1772, -1.1713]` | `±1.0203` | `0.2049` | `unresolved` |
| `g-drift` | 2,400 | `ey1` | `+2.1089` | `+1.0987` | `-0.9655` | `[-1.3197, -0.6266]` | `±0.5272` | `0.1187` | `unresolved` |
| `g-drift` | 2,400 | `ey0` | `-1.9725` | `-1.3046` | `+0.6914` | `[+0.4082, +1.0292]` | `±0.4931` | `0.0872` | `unresolved` |

**One cell of four returns a primary verdict and it is `moved`.** `g-drift` at `n = 600`, gates
passing, `ate` at `-1.7893 [-2.2739, -1.3569]` against a margin of `±0.8997` — the interval wholly
outside the band, in the direction candidate 1 predicts, and the two arm means moving the same way
rather than a contrast cancelling something. Under §8 that is *candidate 1 alive and E2b fires*.

**And one cell is not the study.** The rule is stated per cell and size and there is no combination
across them, so what E2 has is one licensed reading and three unlicensed ones — which is not the
branch this piece was built to take. **Nothing here reads `equivalent`, so candidate 1 is not
dead**; nothing here licenses `moved` in the cell C3c's flat column was most sharply read in, so it
is not established either.

**The unread numbers are large, and saying so is not reading them.** The reference takes `q-drift`'s
column from `+1.1594` to `+0.1114` at `n = 2,400` — about a tenth of what it was, against a margin of
`±0.2899` — and every one of the twelve rows moves in the same direction by several times its own
margin. If those cells' gates had passed, this would be candidate 1 alive by a wide margin. They did
not, and [§8's own worked example](../roadmap.md#what-e2-measured-and-why-it-did-not-branch) is why
that sentence stops there: on the exact law a two-bin reference is wrong about `qr` by more than half
its own magnitude while its `ψ` sits inside a twentieth of a standard error of an exact reference's.
A large paired difference at a reference another resolution beats is a large difference **about the
wrong reference**.

### What it cost

| cell | `n` | companion rows | fits | secs/fit | rounds | invalid | wall clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `q-drift` | 600 | 32,768 | 76 | `19.32` | `9.5` | 2 | 759s |
| `g-drift` | 600 | 32,768 | 76 | `19.28` | `14.5` | 5 | 759s |
| `q-drift` | 2,400 | 32,768 | 76 | `48.39` | `6.7` | 0 | 1,864s |
| `g-drift` | 2,400 | 32,768 | 76 | `45.39` | `14.7` | 1 | 1,757s |

304 fits and about 87 minutes of runner time across four parallel jobs — cheaper than the cost table
sized it for, and cheap enough that the repair below is an errand rather than a decision. The eight
invalid fits are 2.6% of the run and are counted per `(cell, n)` above; `g-drift` at `n = 600` is the
one over the 2% bar, and it is also the cell whose gates passed, so nothing in the licensed reading
turns on it.

### What this dispatch refuses to say

**It does not read the three unresolved cells' differences**, in either direction, and it does not
average them with the fourth. **It changes no constant**: `EQUIVALENCE_FRACTION`, `BUDGET_FRACTION`
and `PRIMARY_ESTIMAND` are what they were before the run and the banner prints them. **It reads no
rate** and makes no coverage claim — item 13 is a rate and closes at E5. **It selects no learner**,
which is E2b's and fires only on a branch this run did not deliver. And it does not move the
concordance's `reduced regressions consistent` row: a gated comparison that could not be read is not
a condition being met.

**What it hands to the next revision is a repair with a name.** §8 already says what a disagreeing
gate-B ordering means — the reference's resolution is not one choice — so the next dispatch is the
same design with **a rung chosen per reduced regression** rather than one for all three, and with the
negative control checked to be rejected on `gr2` at the larger size before anything else is read.
That is a change to the *reference*, which §8 permits before a dispatch with a written reason; it is
not a change to the rule the comparison is judged by.

## What the E2R dispatch measured

*Two dispatches of `.github/workflows/drtmle-reference.yml` at tier 2, no inputs beyond `phase` and
`selection`, on `claude/next-roadmap-step-plan-k2o447`: `--phase select` as run `31084621278` at
`94f3e81d810e7690b5a7d214fa228e1772d98de2`, then the mapping committed as `10289d4f954ff21f1bb20cc270c2496bc305495a`,
then `--phase decide` as run `31087301718` at that commit. Both cells, `n ∈ {600, 2400}`, 16
selection draws and 32 decision draws a cell and size on disjoint data seeds, the rung selected per
`(cell, size, reduced regression)`, four blocks of 8,192 / 8,192 / 8,192 / 2 × 2,048, gate C at 4
scrambles on 4 draws. 76 fits a job, 304 in the decision run; five artefacts,
[manifested](study-manifest.md#e2r-what-was-run) with their digests.*

**The headline is that no cell reaches a readable comparison, and the reason is not E2's.** E2 failed
on one clause — *a coarser rung is strictly better than the shipped one* — in three cells of four.
E2R selected the rung per regression instead, which is the repair E2 asked for, and **that clause now
fails in one cell rather than three**. What holds the other three back is the clause this dispatch
added: a competing rung is **not shown non-inferior** to the selected one within a margin declared
before the run. So the verdict is `unresolved` in four cells of four, and
[the bound applies](../roadmap.md#the-branch-e2r-decides): an `unresolved` E2R ends the
reduction road as evidence rather than earning a third dispatch.

### What the selection cohort chose, and it is not one rung

The selecting run ranked `spline(8)`, `spline(16)` and `spline(32)` on its own 16 draws and its own
block, and `beaten on` reads **0 on all twelve rows** — every regression has an admissible rung and
no cell rests on `FALLBACK_RUNG`.

| cell | `n` | `qr` | `gr1` | `gr2` |
| --- | --- | --- | --- | --- |
| `g-drift` | 600 | `spline(16)` | `spline(8)` | `spline(8)` |
| `g-drift` | 2,400 | `spline(8)` | `spline(8)` | `spline(8)` |
| `q-drift` | 600 | `spline(16)` | `spline(8)` | `spline(8)` |
| `q-drift` | 2,400 | `spline(8)` | `spline(8)` | `spline(8)` |

**The column is not constant, which is the per-regression selection earning its place** — `qr` takes
the finer rung at `n = 600` in both cells and the coarser one at `n = 2,400` — and it is a reading of
*that cohort*. Whether it replicates is the audit's question, and the audit is on draws this table
never saw.

### The gates, which is the table to read first

Every cell is **integrity-valid**: 32 of 32 declared draws on both the paired comparison and the
audit, no fit error, no risk error, nothing missing. All four jobs exited zero, which under
`run_integrity` is the same statement.

| cell | `n` | gate C | gate B | what gate B says |
| --- | --- | --- | --- | --- |
| `q-drift` | 600 | `0.0097` against `0.1196` | **unresolved** | `spline(16)` not shown non-inferior on `h2`, `-7.34e-06` against `-5.93e-06` |
| `q-drift` | 2,400 | `0.0073` against `0.0927` | **fail** | `spline(16)` **beats** `spline(8)` on `qr` |
| `g-drift` | 600 | `0.0345` against `0.3114` | **unresolved** | `spline(16)` and `spline(32)` not shown non-inferior on `gr2` (`-5.24e-04`, `-6.48e-04` against `-1.25e-04`) and on `h2` (`-1.11e-02`, `-5.32e-03` against `-3.95e-05`) |
| `g-drift` | 2,400 | `0.1326` against `0.2732` | **unresolved** | the same two rungs on `gr2` (`-1.09e-04`, `-2.68e-04` against `-5.46e-05`), on `h2` (`-1.15e-04`, `-2.68e-04` against `-7.61e-06`) and on `h3` (`-1.39e-05`, `-1.45e-05` against `-7.43e-06`) |

**Gate C passes in all four**, by a factor of two to thirteen, as it did in E2 — the reference's own
across-scramble spread is not what is holding anything back, and `--reference-points 8192` is where
that came from.

**The negative control is rejected everywhere, on all five metrics.** No cell carries a `why` row
naming a control, which is the clause E2 could not satisfy on `gr2` at `g-drift` `2,400` — there
`bins(8)` read `-1.031e-04 [-4.35e-04, +1.12e-04]`, and here `bins(2)`, `bins(4)` and `bins(8)` are
each rejected with every interval clear of zero. **Clause 4 of the repair took.** So did clause 2's:
the three cells that fail on non-inferiority fail on *competing rungs*, not on the coarsening the
gate exists to detect.

### The one cell that fails outright, and what it says about selecting

`q-drift` at `n = 2,400` is the only cell where a competing rung is **measurably better** than the
selected one, and the direction is the reverse of E2's. E2's failures were the *coarser* `spline(8)`
beating the shipped `spline(16)`; here the selection chose `spline(8)` for `qr` and the audit says
`spline(16)` beats it.

**That is the four-block design answering the question it was built to ask.** The selection cohort's
own table reads `qr = spline(8)` with `beaten on` `0` and the runner-up `spline(16)` at a worst
excess of `+0.0000` — the two rungs are inside the selection block's resolution at 16 draws — and 32
fresh draws on a block neither saw resolve them the other way. **Admissibility on the selecting
cohort did not replicate out of sample.** A run that had certified on the block that chose would have
passed this cell; that is exactly the self-certification clause 2 and the decision protocol were
written against, and it fired.

### Why the other three are `unresolved` rather than passing, and what that costs

The three remaining cells fail **only** the non-inferiority clause, and on the composites the margin
is very demanding by construction. `δ_metric` on `h2` and `h3` is `(FIDELITY_FRACTION · δ)² / (n ·
weight_scale)` — a **squared** tolerance, because Cauchy–Schwarz transfers a mean-square risk to a
mean — so it lands at `5.93e-06` where `q-drift` `600`'s resolvable risk differences are `1e-05` and
its `h2` risks are `1e-03`. The closest miss is that cell: `-7.34e-06` against `-5.93e-06`, short by
about a quarter of the margin on a bound that narrows with draws.

**That is recorded and it is not a repair.** The clause was declared in the commit before the
dispatch, it makes a verdict harder to reach, and a tolerance introduced after a failure is the one
direction a gate may not move. What it does establish is where the instrument's precision sits
relative to what the rule asks for, which is a fact about *this* design at 32 draws and is worth
carrying into any later one.

### What it cost

| cell | `n` | fits | secs/fit | rounds | invalid | wall |
| --- | --- | --- | --- | --- | --- | --- |
| `q-drift` | 600 | 76 | `17.80` | `9.0` | 1 | 716s |
| `g-drift` | 600 | 76 | `23.57` | `9.3` | 3 | 943s |
| `g-drift` | 2,400 | 76 | `43.92` | `13.4` | 0 | 1,712s |
| `q-drift` | 2,400 | 76 | `81.43` | `6.9` | 0 | 3,134s |

Each job carries a companion of `57,344` rows — the four blocks — against E2's `32,768`, and the
decision run's wall clock is its slowest job, 52 minutes. The selecting run is one job over the whole
grid, 64 control-arm draws in 1,538s. **The four blocks are affordable and were not the constraint**;
what a longer run would buy is draws, and the bound is one decision run.

### What this dispatch refuses to say

**It does not read the paired comparison in any cell**, because no cell's gates licence it — the
differences exist in all four and in the same direction, and reading them is exactly what a failed or
unresolved gate forbids. **It does not call E2R `equivalent` or `moved`**: neither verdict was
reached, and `unresolved` is a third verdict rather than a weak one. **It changes no constant** —
`EQUIVALENCE_FRACTION`, `BUDGET_FRACTION`, `FIDELITY_FRACTION`, `COMPONENT_FRACTION`,
`COMPLETENESS_FRACTION` and `PRIMARY_ESTIMAND` are what the manifest recorded at the selection, which
is what `validate_selection` checked before anything was fitted. **It reads no rate and makes no
coverage claim** — item 13 is a rate and closes at E5. **It selects no learner**, which is E2b's and
fires only on a branch this run did not deliver. And it does not move the concordance's
`reduced regressions consistent` row: a gated comparison that could not be read is not a condition
being met.

**What it does settle is the question E2R was dispatched to settle, in the negative.** Candidate 1 —
*the reduced regressions are inconsistent at `glm`* — is neither established nor dead, and **two
dispatches built to decide it have now failed to**, the second on a design repaired at eight points
against the first's own falsifier. The bound the piece set for itself says that is where the
reduction road ends as *motivation for a production change*, and this page says so rather than
asking for a third run.

## What the C3c dispatch measured

*Two dispatches of `.github/workflows/drtmle-coverage.yml` at tier 2, seeds `20250801` and
`20250802`, the second run after the first completed. Both cells, `600 / 1,200 / 2,400`, 250
replicates, `--evaluation-n 2000`, `jobs=2`, on `main` at `0033c82` — the shipped estimator, with
no code change before or between the batches. Runs `30979765029` and `30987423687`; 3,000 draws
and **6,000 fits**; wall clock 4,599s and 6,727s for batch A's two cell jobs and 4,599s and 6,609s
for batch B's, median 2.7s to 4.7s per fit. Per-replicate rows travel as the four artefacts.*

**This is the run [the whole of piece C](../roadmap.md#c-the-demonstration) was built for, and it
is the first of the three attempts that entered the regime it committed to.** Conditions 1 and 2
pass in all four cell-runs: `n^α R₂(Q̄*)` at the plain `TMLE` reads `+0.6063` and `+0.6155` against
a committed `+0.6100` in `q-drift`, `+0.5892` and `+0.6184` against `+0.6200` in `g-drift`. So the
coverage numbers below are about the regime the design named, which is what
[C3a's pilot](coverage-study.md#what-the-pilot-measured) could not say and what
[C3b's repair](coverage-study.md#the-repair-and-what-would-say-each-half-of-it-is-wrong) bought.

### The shortfall, and it is the shape the variant exists for

Coverage of the `ate` interval, pooled over every fit in the cell as §5's fourth rule requires,
with the paired difference and its Monte Carlo error:

| cell | `n` | `TMLE` A / B | `DRTMLE` A / B | `DRTMLE − TMLE` A | `DRTMLE − TMLE` B |
| --- | --- | --- | --- | --- | --- |
| `q-drift` | 600 | 0.716 / 0.784 | 0.792 / 0.820 | `+0.076 ± 0.022` | `+0.036 ± 0.021` |
| `q-drift` | 1,200 | 0.600 / 0.676 | 0.824 / 0.880 | `+0.224 ± 0.029` | `+0.204 ± 0.029` |
| `q-drift` | 2,400 | 0.532 / 0.472 | 0.844 / 0.848 | `+0.312 ± 0.031` | `+0.376 ± 0.033` |
| `g-drift` | 600 | 0.864 / 0.892 | 0.776 / 0.808 | `−0.088 ± 0.023` | `−0.084 ± 0.020` |
| `g-drift` | 1,200 | 0.828 / 0.860 | 0.800 / 0.844 | `−0.028 ± 0.018` | `−0.016 ± 0.017` |
| `g-drift` | 2,400 | 0.712 / 0.728 | 0.780 / 0.784 | `+0.068 ± 0.023` | `+0.056 ± 0.022` |

**`q-drift` is the demonstration and it is unambiguous.** The plain interval degrades as the design
says it must — `√n` bias `+3.07 / +3.62 / +4.29` in batch A, growing at about `n^0.30` against the
`n^0.25` that `α = 0.25` predicts — while `DRTMLE`'s falls `+2.01 / +1.69 / +1.44`. The gain is
`+0.31` and `+0.38` at the largest size, more than six times [gate 2](validation-plan.md#the-decision-rules-frozen-before-the-dispatch)'s
predeclared `0.05`, and the interval on the difference excludes zero in both batches.

**`g-drift` runs the other way at the two smaller sizes, and reproduces doing so.** That is the
cell where the design's own scope statement said `DRTMLE` is *checked to hold nominal under a
drift* rather than claimed to gain, and it does not hold nominal: `0.776` to `0.844` across both
batches, never within reach of `0.95`.

### Nothing in the coverage column is mysterious

Every number above is the Wald coverage implied by that cell's own measured bias and spread. Taking
the half-width as `1.96 × mean se` and expressing both in units of the empirical `mc se`, batch A's
`q-drift`:

| | bias / sd | half-width / sd | implied | measured |
| --- | --- | --- | --- | --- |
| `TMLE` n=600 | 1.30 | 1.94 | 0.737 | 0.716 |
| `TMLE` n=1,200 | 1.66 | 2.03 | 0.642 | 0.600 |
| `TMLE` n=2,400 | 1.89 | 1.89 | 0.501 | 0.532 |
| `DRTMLE` n=600 | 0.89 | 1.85 | 0.827 | 0.815 (excl.) |
| `DRTMLE` n=1,200 | 0.80 | 1.85 | 0.849 | 0.824 |
| `DRTMLE` n=2,400 | 0.67 | 1.77 | 0.858 | 0.844 |

So there is no calibration pathology to look for, in either estimator. What is left to explain is
why `DRTMLE`'s bias stops where it does, and the answer is in the next two sections.

### Why `DRTMLE` stops short of nominal, in the columns rather than in prose

**Its own remainder is not vanishing.** `√n R_remaining`, which is what
[item 13](../roadmap.md#what-is-still-open) is and what Theorem 1 assumes negligible:

| cell | batch A | batch B |
| --- | --- | --- |
| `q-drift` | `+1.427 ± 0.091 / +1.264 ± 0.102 / +1.252 ± 0.139` | `+1.284 ± 0.090 / +1.186 ± 0.099 / +1.174 ± 0.131` |
| `g-drift` | `+4.128 ± 0.175 / +4.117 ± 0.228 / +4.833 ± 0.315` | `+4.043 ± 0.186 / +3.926 ± 0.208 / +4.305 ± 0.334` |

The harness reads `unresolved` in all four, and at 250 draws that is no longer *"not resolvable at
this count"* in the pilot's sense — it is that the quantity is close to **flat**. In `q-drift` the
underlying `R_remaining` falls like `n^(−0.59)`, barely faster than `n^(−1/2)`, so `√n R_rem`
declines about 9–13% over a fourfold `n` against errors of 7–11%. In `g-drift` it does not fall at
all. Two independent seeds agreeing on the shape is better evidence for a plateau than either batch
alone.

**And `DRTMLE`'s bias is descending onto exactly that plateau.** Its `√n` bias in `q-drift` reads
`2.01 / 1.69 / 1.44`, falling at about `n^(−0.24)`; `√n R_rem` sits at `1.25`. Extrapolating the
two, they meet near `n ≈ 4,000–5,000`, after which the bias has nowhere further to fall — which is
`ψ̂ − ψ₀ = (Pₙ − P₀)D* + R_remaining` read as a prediction rather than as an identity. At that
floor the standardised bias is about `0.58` and the implied coverage would be **0.87 to 0.88**,
not `0.95`.

**That extrapolation is exploratory, it is not a ceiling, and this paragraph's own arithmetic is
what says so.** Two reasons, and the second is the one that was missed when the number was first
written down. It rests on a trend the harness itself calls `unresolved` — that much was stated. And
it treats `√n R_rem` as a **floor**, which contradicts the fit three lines above: `R_remaining ~
n^(−0.59)` makes `√n R_rem` decline like `n^(−0.09)`, so there is no level for the bias to land on
— coverage would go on improving, just slowly enough that no reachable `n` shows it. Over a
fourfold `n` a `12%` decline and a plateau are the same picture, and this study cannot tell them
apart. **Nothing else on this page depends on the number and no gate is read from it**; it is kept
because the shape of the argument is worth having, not because `0.87`–`0.88` is a prediction.

**The reported `se` runs short of the actual spread, and it is worth about half the gap.** In
`q-drift` at `n = 2,400` `DRTMLE`'s spread is 4.5% below `TMLE`'s (`0.0443` against `0.0464`) while
its *estimated* `se` is 10.7% below (`0.0400` against `0.0448`) — the `se ratio` of `0.903`, which
reproduced to the digit in batch B. `σ²ₙ` is the empirical variance of the estimated curve and
treats the reduced regressions as known, so their estimation error is in `ψ̂`'s spread and not in
the variance estimate. Give `DRTMLE` a correctly sized interval at that size and the same residual
bias yields `0.898` rather than `0.844`: of the eleven points to nominal, about five are the `se`
and six are the remaining bias.

> **Two sentences of that paragraph are withdrawn, and the measurements in it are not.** The
> `0.0443` / `0.0400` / `0.903` are what they were and reproduced to the digit; the `0.898` is
> reproducible to four places from `Φ(1.96r − b) − Φ(−1.96r − b)` at `r = 0.903`, `b = 0.664`.
> What is withdrawn is the *mechanism* — `σ²ₙ` "treats the reduced regressions as known" describes
> **Theorem 1's own variance formula** rather than a defect: an omitted non-negative term predicts
> an `se ratio` below one in every cell, and `g-drift`'s reads `1.157`. And the *split*, which is an
> artefact of applying the two counterfactuals in one order: the same arithmetic run bias-first
> gives `0.9232` and leaves `+0.027` for the `se`, and the two gains sum past the gap they are
> dividing. [The roadmap carries the
> withdrawal](../roadmap.md#the-se-shortfall-is-a-symptom-and-what-withdrawing-it-costs) and the
> table. **No gate verdict moves**, and the reading below this heading is untouched.

### Why `g-drift` is the cell it is, and it is one column

**The correction removes almost nothing there.** The regime-entry table at `n = 2,400`:

| cell | `TMLE` `n^α R₂(Q̄*)` | `DRTMLE` | `within` |
| --- | --- | --- | --- |
| `q-drift` A / B | `+0.6155` / `+0.6063` | `+0.1852` / `+0.1817` | `0.30x` / `0.30x` |
| `g-drift` A / B | `+0.6184` / `+0.5892` | `+0.6020` / `+0.5686` | `0.97x` / `0.92x` |

`DRTMLE` removes 70% of the targeted remainder in `q-drift` and 3–8% in `g-drift`. The remainder
decomposition says the same thing from the other side: in `g-drift` `R_Q` is `+0.098` against
`R_g`'s `−0.032` at the largest size, so what survives is the **outcome-side** reduction, and
`g-drift` is the cell whose outcome regression is the deliberately misspecified GLM. At `n = 600`
that leaves `DRTMLE` carrying a *larger* bias than `TMLE` (`+0.1260` against `+0.1186` in batch A)
inside a *narrower* interval (`0.4472` against `0.4647`) — two measured quantities pushing the same
way, which is why the negative sign reproduced rather than washing out.

**`cancel` is the column that moves most across sizes here**, `1.00x → 1.07x → 1.42x` in batch A
and `1.00x → 1.21x → 1.99x` in batch B: the two appendix branches increasingly oppose each other as
`n` grows, so gate 1's clause 4 fails on its *second* half in this cell as well as its first.
`branches settled` also falls to `192/250`, so the binned limits were still moving between the two
bin counts at the largest size. That is a **stability** count and not a resolution one, and
`cancel` is withdrawn as evidence for the clause — [the gate
readout](coverage-study.md#the-gates-read-out-clause-by-clause) says why the verdict is unchanged.

### What did not go wrong, and it is worth stating plainly

**Zero state-identity failures across all 6,000 fits**, in every cell, at every size, in both
batches. [B1a](../roadmap.md#b1a--the-identity-and-safety-patch)'s distinction between a software
defect and a fit that did not converge is doing exactly the work it was worded for: every invalid
fit in this study is a `score` failure. The rates, `DRTMLE` only — `TMLE` never records one —
against the 2% threshold frozen after the pilot:

| cell | batch A | batch B |
| --- | --- | --- |
| `q-drift` | `0.028 / 0.008 / 0.012` | `0.032 / 0.028 / 0.008` |
| `g-drift` | `0.072 / 0.052 / 0.032` | `0.060 / 0.036 / 0.028` |

Ten of the twelve are over the bar, and the bar stays where it was — [that
paragraph](validation-plan.md#four-rules-that-make-the-gates-operational) exists to say a threshold
is not moved to the number that was seen. The rate falls with `n` in both cells, which is the
reduced regressions getting better, and it is the gap between the primary and excluded coverage
columns: `0.776` against `0.836` at `g-drift`'s smallest size.

**The cells are mixed and every one of them reads `BOUND-ACTIVE`**, at `1.2%` to `8.8%` of draws —
`q-drift` `6/5/4` and `15/8/3`, `g-drift` `11/18/11` and `22/13/17` out of 250. The initial
mechanism's clip share is `0.0000` to `0.0017`, so this is C1's finding at 250 draws rather than
six: the tilt reaches a bound where the outcome regression is right, not where overlap is poor. The
strata are reported beside the pooled number as description and neither is quoted as a verdict.

## What the sizings got wrong

Twenty-two lessons, distilled from the per-item retrospectives that used to run to several hundred
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

**13. A stop-ship's stated *reason* can carry the very error the stop-ship was written to catch,
and only building the thing it talks about will find that.** [Stop-ship
14](../roadmap.md#stop-ship) exists so that `test_influence_gateaux_drtmle`'s agreement is not read
as evidence about the cross-fitting construction. Its conclusion was right for two revisions. Its
reason — "it runs at saturated reductions, where every conditioning cell is a singleton" — was
wrong twice over, and so were the two other documents that repeated it: on that law the design
takes three values over a thousand rows, and saturation of the *reduction* does not decide the
question at all, since under a primary learner that learns any reduction learner returns different
arrays. What makes the module silent is `cross_fit=False` **and** oracle primary learners.

The damage a wrong reason does is not cosmetic, and it is the specific thing worth remembering.
A conclusion protects only the case it was written about; a *reason* is what a reader generalises
from. This one would have licensed reading a **cross-fitted** saturated fit as evidence about fold
reuse — precisely the inference the clause forbids — so the guard-rail was pointing at the wrong
hazard while looking like it pointed at the right one.

It survived because nothing could fail against it. The claim was about a construction that did not
exist, so no test could contradict it, and it read as a technical detail rather than as a claim
anyone would check. What found it was [A1b](../roadmap.md#a1b--the-cross-fitting-construction)
building the nested construction and asking what the pooled one had to agree with it *about* — at
which point the singleton claim did not survive its first contact with the fixture. **The rule:
when a document says an instrument is blind, the reason is a claim like any other, and the way to
check it is to build the thing it says the instrument cannot see.** The corrected statement is now
asserted rather than described, and is kept as a mutation watched to *pass*.

**14. A column that reproduces its own quadrature exactly can still be about the wrong quantity,
and being exactly right is what stops anyone checking.** C3's pilot found Tier 1's regime-entry
column reading `n^a R2 = +0.4000` at every size — the declared coefficient, to four decimals — beside
a plain `TMLE` that did not under-cover anywhere, against a design predicting a shortfall of `0.08`
to `0.14`. Nothing was wrong with the arithmetic. `exact_remainder` integrates the **plug-in**
remainder at the initial regression, which its own docstring says; a fit's bias is the same
expression at the **targeted** one, and the fluctuation's score equation constrains the second
while leaving the first alone. Measured on the same rows of the same fits, the bias tracked
`R2(Qbar*)` at `-0.004`, `+0.011`, `-0.002` while `R2(Q-hat)` sat at `+0.081`, `+0.068`, `+0.057`.
A test suite that checked the column against its quadrature — which it should — could not have seen
it, because both sides of that check were the same quantity.

**C3b sharpened the size of it and the sharpening is its own small lesson.** *"A factor of twenty"*
was what the pilot could see, and it was a **noise-floor artefact**: at 24 draws the measured
`R2(Qbar*)` was consistent with zero, so the pilot had bounded the ratio rather than measured it.
Computed exactly, `b_ATE = 0.00092` against `c_ATE = 0.40` — a factor of **436**. A ratio read off
a column that is statistically indistinguishable from zero is a lower bound on the ratio and
should be written as one.

**The generalisable half is which check was missing.** Two arms were compared and agreed: the
column and the analytic coefficient. What was never compared is the column against the thing it was
*used for*, which was a prediction about coverage. This is [lesson 9](#what-the-sizings-got-wrong)'s
shape in a new place — two checks that cannot fail against the same class of error are one check —
and the instrument that broke the tie, `benchmarks/drtmle_tier1_bias.py`, is thirty lines and runs
in seconds. **When a design predicts a number, check the prediction and not only the input to it.**
[The validation plan's §5](validation-plan.md#verifying-the-regime-was-entered) now requires the
targeted coefficient as a pre-flight condition for exactly this reason.

**15. The diagnosis said "project the absorbed component out"; the algebra said "solve for a second
coefficient", and those are different pieces of work.** C3a's own repair section proposed
projecting out the direction the fluctuation reaches and renormalising, and warned — rightly — that
a one-dimensional projection removing 95% of a quantity might be treating a symptom. Writing the
elimination out instead of describing it gives `b_a = P_0[v_a h_a]` against a computable weight:
**a linear functional of the injected shape, exactly as `c_a` is**. So the repair is a 2x2 Gram
solve for two declared coefficients, the old design is its one-condition special case, and there is
no projection anywhere. The distinction is not stylistic — a projection has no declared target, so
it could not have been checked against one, and the whole point of the repair is that a pre-flight
now has a number to read against.

**Two things fell out of writing it that describing it would not have found.** Declaring `b = c`
forces `P_0[w_a h_a] = 0`, so the injection is exactly orthogonal to the score and the fluctuation
absorbs nothing — an identity, not a tuning. And the design's opposite-arm signs, which make
`c_ATE` a sum of magnitudes and cancellation impossible, **do not carry over to `b`**: both targeted
arm coefficients came out positive, so `b_ATE` had been a *difference*. The no-cancellation
guarantee was being enforced on the column that is not the estimand's, which is lesson 14 one level
down. **When a diagnosis names a fix, derive the fix before costing it.**

**16. A repair's obvious knob is worth one measurement before it is worth an argument.** Tier 2's
realised coefficient came in at `1.5`–`1.6x` its prediction, and the prediction is the `h^2`
leading term alone at a bandwidth of `h(600) = 0.517` — so the excess reads as an `h^4` truncation
error and the fix reads as a smaller `c_h`. That is a clean story and it is wrong. Scanned over
`c_h` of `1.15 / 1.00 / 0.90 / 0.80 / 0.70`, the ratio goes `1.61 / 1.78 / 1.91 / 2.05 / 2.21`: it
**rises** as the bandwidth falls, which no bias-side omission does. The omitted term is
variance-side — both nuisances are fitted on the same rows and their errors covary — so no
bandwidth makes the leading-order prediction correct and shrinking it makes the agreement worse.
The scan is forty lines and eight draws a cell. **The cost of checking a knob is usually below the
cost of writing the paragraph defending it.**

**17. A regime-entry condition read on one estimator says nothing about the other, and this is the
third instance of the shape lesson 14 named.** The pre-flight's conditions 1 and 2 are read on the
plain `TMLE` **by construction** — the design commits `TMLE`'s remainder, and that is the interval
a shortfall is claimed against — so all four cell-runs passing them says the *plain* estimator is
in the designed regime. It says nothing about whether the *corrected* one is in a regime where its
own guarantee holds, and in `g-drift` it is not: the entry column reads `0.92x` to `0.97x` for
`DRTMLE` against `TMLE`'s `1.00x`, meaning the correction removed 3–8% of what it was supposed to
remove. The two columns sit side by side in the same table and one of them was the pre-flight and
the other was not. **A verdict table with a per-estimator column is only a verdict about the
estimator whose row is being read**, and the fix here is not a new instrument — both numbers were
already printed — but reading the second one before believing the first.

**22. A componentwise loss is not a loss on the thing the estimator divides by, and the difference
lives entirely in a weight.** E2 gated its reference on three held-out risks, one per reduced
regression, which is what Theorem 1's premise is stated in — and none of the three is what a fit
reads. `reduced_correction_parts` builds `q_r/g*` and `g_{r,2}/g_{r,1}` at the *bounded* denominators,
so the same estimation error matters more where the estimator divides by something small, and a
ranking on the numerator can order two candidates the other way from a ranking on the ratio. **The
fix costs one array and no new theory**, because the property the gate rests on survives it: a
held-out risk's cross term vanishes for any weight measurable in the conditioning index, and both
divisors are functions of that index — `q_r`'s index *is* the mechanism it is divided by. So the
composite loss is the component loss under `w/d²`, with the irreducible term still common to every
candidate, which a *ratio* of two risks would not have been. Two things have to be said with it. The
divisor must be **the fit's own**, so it is one array for every candidate rather than each
candidate's own, or the difference of two risks becomes a difference of two irreducible terms. And
the composite is then blind to the divisor's *own* error, which is what keeps the componentwise
risks in the gate rather than replacing them: **componentwise risks are theorem-relevant and
incomplete, not wrong.** The general form: when a gate scores a component of a composite, ask what
weight the composite implies and whether the loss's own algebra survives it.

**21. A ranking becomes a selection the moment you act on it, and then neither the block nor the
state it was ranked at can certify it.** E2 shipped one rung and used a held-out block to check it;
three cells of four failed that check on *another rung being better*, and the obvious repair — choose
the rung against the ranking — quietly voids the check, because whichever rung wins on a block wins
there by construction. So the block splits: one to choose on, one to certify on, from disjoint
scramble streams, and a run that reports both is a run where "won there and lost here" is visible
rather than inferred. **The half that is easy to miss is that the *state* is part of the instrument
too.** These candidates are regressions on a fitted nuisance, so the state fixes the conditioning
index and the divisors — and a rung chosen at the initial pair is chosen at a mechanism the
alternation has not yet made bound-active, which is a divisor no fit uses. A six-draw pilot ordered
the rungs one way there and the other way at the exit state. The state that works is the **control
arm's** exit: candidate-free, since the control is the comparison's other arm and not a rung, and
targeted, so the selection and the audit read the same kind of divisor. What cannot be had is the
state the certification happens at, and saying so is the point of having an audit at all.

**And the third thing to carry over is the statistic**, which is the half that cost a rewrite. The
first selection rule minimised each rung's worst *relative excess* of the mean risks — the natural
minimax, and judged by a quantity the gate does not use. On one metric it bought a `2e-06` loss
that six draws **resolved**, interval clear of zero, for an apparent `1e-05` gain on a metric whose
intervals straddled zero by an order of magnitude; it selected a rung the gate then rejected on
precisely the difference the rule had discounted. **A selection judged on point estimates cannot be
certified by a clause read on intervals.** The repair is to select on the gate's own statistic —
the coarsest rung that no other rung *significantly* beats — which turns the audit into a
replication test rather than a coin flip. The general form: when a diagnostic becomes a selector,
re-ask all three of its independence questions — which rows, which fitted state, and which
statistic.

**20. A successive difference is a stability diagnostic and an independent randomisation is an
error estimate, and reading the first as the second is the most natural mistake in numerical
work.** E1 read a nested Sobol ladder's `delta` — the movement between a rung and the next coarser
one — as bounding the grid's remaining error, and shipped "one to three per cent of the column's
spread" on it. A small successive difference bounds nothing without monotonicity or a convergence
result that applies, and the same module's docstring had already recorded that Tier 2's integrand
is only piecewise smooth so Sobol's rate is not guaranteed — the argument for the retraction was
sitting four paragraphs above the claim. Measured on that ladder's own geometry, the finest rung's
`delta` ran **four times below the actual error** on a piecewise-smooth integrand, and three orders
*above* it two rungs earlier: it is not conservative, it is uninformative. What the rule's error
needs is replication rather than refinement — independent scrambles of the same quasi-random rule,
which assume no rate at all — and randomising the scramble *per replicate* additionally makes the
error mean-zero, so it stops being a bias a study cannot average down. **Both fixes were one
keyword argument away and neither was reached for, because the ladder looked like evidence.** The
general form: when a diagnostic is a difference between two values of a tuning parameter, ask what
it would read if the sequence had converged to the wrong answer.

**19. An error budget written from an assumed constant is not a measurement, and the measurement
is usually a subtraction of two spreads you can already produce — but check what the subtraction
identifies.** The record here said a replicate's `P₀D̂` carried `sd(D)/√m ≈ 0.023` at `m = 2,000`,
computed from an assumed `sd(D) ≈ 1.0`, and that number was quoted for three revisions as the
reason a single row's column was mostly noise. It was the right shape and the wrong size: `D̂` on
this law has a standard deviation nearer `1.7`, so the true figure is `0.033` to `0.041`. That half
stands. **The corollary this lesson drew from it does not, and it is the second half of lesson 20.**
It said differencing the two rules' measured spreads *is* the decomposition, and a difference of
two variances identifies a variance only when the shared component is uncorrelated with both error
terms. Here one of them qualifies and the other does not: the i.i.d. companion is drawn
independently of the fit, so its error is mean-zero given the fit; a *fixed* grid's error is a
deterministic function of the fitted curve. **A subtraction is a decomposition only under an
independence you have to state.** The diagnostic point in the original lesson is untouched and
worth keeping: the first witness written for this was the movement when half the rows are dropped,
which is a fair reading of a *bias* and a `1.4x` overstatement of a *noise* — it reported the
i.i.d. rule as more than 100% of its own variance, which is not a share of anything. A derived
diagnostic carries the model it was derived under; two measured spreads carry a subtler one.

**18. Producing the gap and clearing the gate are different results, and a study can do the first
while failing the second.** Three attempts found no gap; this one found a large, reproducible one —
`+0.31` and `+0.38` paired at the largest size, six times gate 2's threshold — and `DRTMLE` still
does not clear gate 1, because its *own* remainder does not vanish, its interval never reaches
nominal, and it fails to converge in 1–7% of draws. Those are not in tension: the variant removes
most of the plain estimator's bias and is left with a second-order term that the theorem assumes
away and these sample sizes do not deliver. The temptation a "success" creates is to lead with the
`+0.38` and let the four failing clauses become caveats. **Read the gate clause by clause against
the columns that were frozen for it, and report the verdict the clauses give rather than the one
the headline number suggests.**
