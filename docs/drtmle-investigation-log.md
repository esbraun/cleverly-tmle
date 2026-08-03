# DRTMLE: the investigation record

What was measured, what was hypothesised and dropped, and what one execution environment could
and could not reach. **This file is a record and not a plan** — [the roadmap](roadmap.md) is the
plan, [the concordance](drtmle-theorem-concordance.md) is what the sources say, and [the
validation plan](drtmle-validation-plan.md) is what will be run. Everything here is kept for the
same reason `bench_tmle.py` keeps its own numbers: a measurement nobody can rerun becomes
folklore, and a hypothesis that was dropped without its reason written down gets proposed again.

It exists because the roadmap had become a status page, a methodology review, a forensic report,
an implementation spec, a test plan, a simulation protocol and a project history at once, and the
[second external review](#reviews-received) was right that the density made the dependency plan
harder to execute than the plan itself was.

## Reviews received

Two, both kept in full and neither adopted unread.

- **[The first review](drtmle-review.md)**, verbatim, read the plan against Theorem 1 of
  Benkeser et al. (2017) and found the definition of done right and the route to it short by two
  conditions — now items 13 and 15. Three of its charges came back narrower than stated when
  checked against the code (its §1 terminology charge, §3's monotonicity charge, §7.1's on
  weights); the roadmap records each with what survives of it.
- **The second review**, of 2026-08-02, read the *revised* roadmap — the one that had turned the
  first review into a dependency-ordered execution plan — plus the 2016 Berkeley working-paper
  version of Benkeser, Carone, van der Laan & Gilbert. It accepted the item-20 diagnosis below,
  rejected three of the revision's claims, and supplied the theorem objects that
  [the concordance](drtmle-theorem-concordance.md) is now seeded with. Its findings are
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
rather than from what the solver recorded. That is [item 16](roadmap.md#closed-since-this-list-opened)
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
- **It is not [limitation 5](roadmap.md#limitations-recorded-rather-than-fixed).** Those two
  looked like one story — "four to five orders worse on a quarter of draws" — and are not. The
  `1e-9` of limitation 5 is the equation the loop poses, measured at the arrays the loop leaves,
  and it stays `1e-9` on the uncentred draws too.
- **It is not [limitation 6](roadmap.md#limitations-recorded-rather-than-fixed) either.** The
  closing pass's mechanism stage does bind on its cap, and the uncentred draws are the ones where
  the tilted `g*` leaves the bounds — but the cap binds on 94 of 96 fits while the curve is
  uncentred on a quarter of them, so the cap cannot be what selects them.

### What B1a measured when it landed

The patch is described in [the roadmap](roadmap.md#what-b1a-landed); what belongs here are the
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
is [item 23](roadmap.md#what-is-still-open), a single-guard fit subtracting a correction it never
solved for. It is a different defect from item 20 in cause, in magnitude and in which fits it
touches, and nothing here would have seen it: no test in this repository fits a partial guard end
to end, and the default `guard=("Q", "g")` cannot be in this state.

That is [lesson 8](#what-the-sizings-got-wrong) arriving from the other side. Item 20 was found by
recomputing a recorded number from the returned state on one fit; this was found by the same
recomputation, once it was a permanent fixture of every doubly-robust fit rather than thirty lines
in a scratch file.

**Item 23 is now [closed](roadmap.md#closed-since-this-list-opened)**, and the measurement above is
kept because it is the evidence rather than a symptom. The curve subtracts one correction per
equation the guard asked for; the unsolved equation's mean is still reported, as a diagnostic held
to no threshold, which is the row that found this. On the 600-row draw
`tests/unit/test_drtmle_fit.py` fits everything else on, the two arms read `1.2e-03` and `3.1e-04`
against a `5.4e-06` bar — the same finding on a second draw, and now the fixture of the first
partial-guard fit this repository has ever run end to end.

### What the B1b prototype measured

Sizing for [piece B1b](roadmap.md#b1b--the-theorem-conforming-targeting-decision), which has since
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
is [stop-ship 14](roadmap.md#stop-ship)'s shape — a check agreeing where it could not have
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

**These numbers measure the exit criterion [item 7](roadmap.md#closed-since-this-list-opened)
replaced**, not the current one. That is deliberate and is the order the item required — the
failure had to be characterised before the threshold moved — and re-measuring under the current
rule is [piece B2](roadmap.md#b2--the-sweep-on-the-corrected-implementation)'s.

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
— the prediction [item 4](roadmap.md#limitations-recorded-rather-than-fixed) makes and had never
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
`docs/pdf.pdf` is Benkeser & Hejazi (2023), *Doubly-Robust Inference in R using drtmle*,
Observational Studies 9(2):43–78, and the 2016 Berkeley working-paper version of Benkeser, Carone,
van der Laan & Gilbert was read by the second review and transcribed into
[the concordance](drtmle-theorem-concordance.md). **Check the network again rather than inheriting
this measurement**, and prefer a checked-in copy to either.

## What the sizings got wrong

Eight lessons, distilled from the per-item retrospectives that used to run to several hundred
lines. They are kept and the retrospectives are not, because the only thing a retrospective is
for is the next sizing — the full pre-work read of what `drtmle` would touch, the per-seam record
of what each cost, and the six landed refusals' own notes are in git history, last carried in full
at `da8cacf`.

**1. A refusal's stated reason is the first thing to check, and it is wrong about half the time.**
Three of the six lifts in [Refusals worth lifting](roadmap.md#refusals-worth-lifting) found the
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
code, three of [the first review](drtmle-review.md)'s charges came back narrower than stated (14,
17 and 19, each still real) and the rest came back whole. That ratio is about the same as lesson
1's on refusals, and for the same reason: a written justification is a claim with no instrument,
so it decays at the rate claims with no instrument decay. **The cheapest instrument for a prose
claim is a reader who has the source open**, and one pass of that over this page cost less than
any item on it. The second review is the same lesson arriving with the *source itself* attached,
and it found [item 21](roadmap.md#a1a--the-theoretical-audit) — a sign — which no reader without
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
[B1b](roadmap.md#b1b--the-theorem-conforming-targeting-decision) eventually adopts.

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
the value the check has to be taken at.** [A1a](roadmap.md#a1a--the-theoretical-audit) was sized
as a further `discrete_law*` module carrying the whole DRTMLE limit as an analytic functional:
initial nuisances, three reduced regressions, the alternation, differentiated by complex step.
That is a few hundred lines, it re-encodes the algorithm rather than the derivation, and it would
have had to reproduce a loop [limitation 4](roadmap.md#limitations-recorded-rather-than-fixed)
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
14](roadmap.md#stop-ship) exists so that silence is not later read as agreement. And it made three
mutations invisible, each found by running it and watching it *pass*. That half of the record is
the one that is never kept: a suite documents what it caught, and what it cannot catch is what a
later reader mistakes for coverage.
