# DRTMLE: the validation plan

What will be run, on what fixtures, against what frozen decision rules. [The
roadmap](../roadmap.md) says which pull request lands which of these and in what order; this file is
the detail those pull requests are executed from, so that a rule is written down before the number
it judges exists.

Six sections match six pieces of work: [B1a](#1-the-invariants-piece-b1a) is the identity patch,
[B1b](#2-the-targeting-candidates-piece-b1b) is the targeting decision,
[A1a](#3-the-component-checklist-piece-a1a) is the component checklist,
[B2](#4-the-sweep-piece-b2) is the convergence and overlap sweep,
[C](#5-the-controlled-study-piece-c) is the demonstration, and
[A1b](#7-the-cross-fitting-construction-piece-a1b) is the cross-fitting construction. [The mutation
table](#6-what-each-new-test-has-to-be-watched-to-fail) is what makes any of it evidence.

§7 sits after §6 rather than before it because §6 was already numbered when A1b's section was
written, and renumbering a page every document on this site links into by section number costs more
than an out-of-order heading does.

§3 used to be `drtmle` parity. That piece is
[retired](../roadmap.md#closed-since-this-list-opened) — no R here and none in CI — and what took its
place is the same decomposition checked against the derivation instead of against another
implementation.

## 1. The invariants (piece B1a)

**Landed.** [The roadmap](../roadmap.md#what-b1a-landed) says what shipped and which four decisions
inside it were forks; this section is the specification it was executed from, kept as written
except where a number in it was wrong — see [the sign](#the-clipping-diagnostic) below.

**This lands before any convention is chosen and is valid under every one of them.** Its purpose
is to make [item 20](investigation-log.md#item-20-from-discovery-to-cause)'s class of
defect impossible to hide, not to decide what the mechanism should be.

### The identities

For each arm `a`, computed **from the exact returned state** and nothing else:

```text
S_g(a) = Pn[ w · D*_g(a) ]
S_Q(a) = Pn[ w · D*_Q(a) ]
D̂_DR(a) = D̂*(a) − D̂_Q(a) − D̂_g(a)          (rowwise, sign per the concordance §4)
```

with the residuals the fit must expose:

```text
Δ_g(a) = S_g^stored(a) − Pn[ w · D*_g(a) ]
Δ_Q(a) = S_Q^stored(a) − Pn[ w · D*_Q(a) ]
```

Five conditions on how they are checked, each ruling out a way of passing for the wrong reason:

- **per arm**, never only on the ATE — arm-specific errors cancel in a difference, and the ATE
  curve is the rowwise difference of the arm curves;
- **before** the contrast is constructed;
- with the **row weights** included, since every score here is weighted;
- on the **same outcome scale** — `OutcomeScaler` sits between the equation's scale and the
  report's and is exactly the second instance of
  [lesson 8](investigation-log.md#what-the-sizings-got-wrong)'s pattern;
- on a fixture **where the truncation binds**. A fixture where it never binds passes either way;
  this is the degeneracy that hid the defect in the first place.

### The clipping diagnostic

The exact discrepancy, exposed as a number rather than inferred:

```text
B_clip(a) = Pn[ w · Q_r(a, W)/g_b(a|W) · { g_raw(a|W) − g_b(a|W) } ]
```

On the current implementation this must reproduce the mechanism score/correction mismatch to
floating-point, which is what makes it a check on the diagnosis and not only a new column. It
stays useful afterwards: under any convention it measures how much of the mechanism equation the
bound is absorbing — though **not as a fixture selector**, because after
[B1b](#2-the-targeting-candidates-piece-b1b) it is zero at the exit of every fit that carries the
bounded array forward. What it says then is that the fix took; what it can no longer say is that
this draw was a hard one.

**It does, and the sign above is one orientation out from the residual it explains.** With
`B_clip` carrying `g_raw − g_b` as written here and the residual defined as
`Δ_g = S_g^stored − Pn[w D*_g]` as written above, the two are *negatives* of one another:

```text
Pn[ w · D*_g(a) ] − S_g^stored(a) = Pn[ w · B_clip(a) ]
```

because one residual reads `1_a − g` and the other `1_a − g^b`. Both are kept as specified —
`CorrectionParts.clip_bias` is this document's `B_clip` and `CorrectionRow.residual` is this
document's `Δ` — and the relation is asserted with its sign rather than being tidied away. It
holds to floating point, per arm, on a draw that clips: `Δ_g = 3.410e-03` against
`B_clip = −3.410e-03` at arm 0 and `−2.449e-04` against `2.449e-04` at arm 1, on the
`repeats=2` fixture's draw 1.

### What the fit then does

Mark inference **invalid** whenever any identity residual exceeds numerical roundoff, or any final
correction score **for an equation this fit's guard solves** exceeds the predeclared inferential
tolerance, and say so on the face of `summary()` — the machinery is
[item 16](../roadmap.md#closed-since-this-list-opened)'s and already exists. The two conditions are
different failures and must not be reported as one: the first is a software defect, the second is a
fit that did not solve its equations.

The qualification is [item 23](../roadmap.md#closed-since-this-list-opened)'s and is not a loosening:
a correction whose equation the fit never posed is not in that fit's curve, so it cannot invalidate
an interval however large it is. It is still *reported*, as a third row kind (`diagnostic`) held to
no threshold, because it is the only thing that says what a single guard did not buy — and because
it is what found item 23.

*"Numerical roundoff" is now a number*: `validation.drtmle.IDENTITY_TOLERANCE = 1e-12`, absolute,
on the outcome scale, and deliberately not relative to the score — a difference between two
evaluations of one expression has zero as its right value rather than something small compared
with anything. Measured on `nonlinear_dgp`, a holding identity sits at `2e-19` and the smallest
real failure at `7e-08`. The inferential tolerance is `score_check`'s own
`DEFAULT_TOLERANCE · se/√n`, so the two rows are held to the two bars they should be. Both kinds
appear in `score_check` as their own row kinds (`identity`, `correction`), which is what puts them
in `summary()`; `res.validation.correction_check()` is the recomputation behind them.

`tests/unit/test_drtmle_fit.py::TestTheReportedCurveIsNotAlwaysCentred` pinned the defect's numbers
and was to be **rewritten rather than deleted** — its fixture is the regression test that the bound
still binds on that draw, which is the thing every later assertion needs. It has been rewritten
twice: by B1a, to assert the identity rather than the symptom, and by B1b, which made the identity
hold and renamed the class for what it now says. Both halves of the claim still live in one fit,
its draw 0 having been the one where the bound never bit and its draw 1 the one where it did. A
fixture chosen for passing would prove nothing and one chosen for failing would prove little more.

## 2. The targeting candidates (piece B1b)

**Landed, and D is what was selected.** [The roadmap](../roadmap.md#what-b1b-landed) says what
shipped and which four decisions inside it were forks; this section is the specification it was
executed from, kept as written except where the prototype corrected it — the candidate table's
axis, and the fixture witness at the end.

**There were more than two conventions and the defect could be removed under any of them.**
This is the correction the second review is most insistent on and it is right: matching R would
make the recorded score and the reported correction refer to one expression, and it would *not*
make hard clipping after a logistic fluctuation solve that expression's score equation.

| variant | residual | denominator | update | what it is for | measured |
| --- | --- | --- | --- | --- | --- |
| **current** | raw | bounded | logistic | the baseline defect | `Δ_g` up to `3.7e-03` |
| **A — post-fit clip** | bounded | bounded | logistic, then clip, then iterate | one array, internally consistent | identity holds; final score `6.8e-06` where the bound binds at the fixed point |
| **B — raw throughout** | raw | raw | logistic | the literal unbounded score | not run — unusable as a default |
| **C — hybrid** | raw | bounded | logistic | the score Python actually solves today | **current** with the curve made to follow it |
| **D — direct bounded** | bounded | bounded | root / Z-solve | the exact bounded equation | **selected**: identity holds; final score `2.1e-10` on the same fit |

**A column, and two rows that are one solver.** The measurements are [the prototype's in the
investigation log](investigation-log.md#what-the-b1b-prototype-measured), and two things
came back that this table did not anticipate. **C and current pose the same equation** — the rows
are identical across the first three columns, and what would change in adopting C is the *curve*,
made to read the raw residual so that it follows the solver. That is a real option and it is the
one criterion 1 rules out first, since a `D_g` whose residual and denominator sit at two different
mechanisms is not a term any theorem here derives. And the axis this table is organised by, which
mechanism each expression reads, is not the axis that carries the defect: that is **which array the
alternation carries forward**, since `targeted_g` is the raw tilted mechanism and the next round
offsets from its `logit`. Carrying the bounded array forward makes the identity hold at the exit
under both A and D, because at a fixed point `ε → 0` and the two arrays coincide there.

**A — post-fit clipping, as `drtmle` does it.** `fluctuateG` applies `pred[pred < tolg] <- tolg`
to the fitted values and returns *that* as `gnStar`, so R has one array and cannot be in the
current state. That is **internal consistency and not a theorem**, and it is the whole of the
argument for this candidate now that reproducing R is not a goal: it eliminates the mismatch and
is straightforward at the `DRTMLE` call sites. Against it: each logistic substep solves the
**pre-clipping** score, so convergence rests on the outer iteration rather than on a substep
identity, and an exact root may never be
reached while observations sit on the boundary. R sidesteps this by capping at three iterations
and never claiming convergence.

**B — raw everywhere**, treating a positivity failure as an invalid fit rather than modifying the
equation. Equation (9) keeps its literal untruncated form and the logistic fluctuation keeps its
ordinary score identity. Against it: numerical instability at small probabilities, and it may
violate the boundedness the asymptotics assume — unusable as a default where practical positivity
is poor.

**C — bounded denominator, raw residual.** Exactly matches the score Python solves today and
stabilises the covariate. **It is not theoretically neutral**: it is not the theorem's `D_g(g)`,
because the denominator and the residual are then evaluated at *different* mechanisms, and it
needs its own drift derivation. Matching an optimiser's score is not the same as licensing the
corrected influence function.

**D — solve the bounded estimating equation directly.** Define

```text
F(ε) = Pn[ Q_r/g^b_ε · { 1(A = a) − g^b_ε } ],      g^b_ε = clip(expit(logit g + ε·H_g))
```

and solve `F(ε) = 0`, rebuilding the covariate as required — by scalar root finding, a constrained
estimating-equation solver, a **smooth** bounded submodel derived so its likelihood score is the
desired equation, or an active-set/KKT formulation if hard bounds stay. It targets the expression
inference actually uses and separates theorem conformity from R's numerical convention. Against
it: hard clipping makes the map non-smooth, a zero need not exist inside the bounded set, and a
smooth submodel needs a fresh derivation and may no longer be a standard likelihood fluctuation.

**The four routes in that sentence are not four candidates**, and the prototype narrowed them to
two, measured them against each other, and **D-hard is what landed** — `clip` inside `F`, the
pinned rows contributing nothing to the Jacobian, the root found by `scipy.optimize.root`'s `hybr`
with this package's own convergence verdict on top. **D-smooth** —
`g_ε = lo + (hi − lo)·expit(logit((ĝ − lo)/(hi − lo)) + εH_g)`, so the mechanism cannot leave the
bounds and nothing is projected — is what [the concordance's
§7](theorem-concordance.md#7-truncation-is-not-in-the-theorems-algorithm) reads as
preferring, and it lost on both criteria that separate them. It is a *different submodel on every
fit*: at inert bounds of `1e-6` it moved a no-clip fixture's `psi` by `2.7e-03` standard errors
where D-hard moves it by zero, which would break the `1e-12` window of the one module whose point
is that tolerance. And where the bound binds it left the final score at `1.5e-07` against
`2.1e-10`, its derivative `(hi − lo)·u(1 − u)` collapsing near the bounds. §7's preference is an
argument against a projection applied *after* an unconstrained optimisation, which is candidate A;
D-hard puts the clip inside the equation, so the stated reason does not reach it.

### The decision hierarchy

**Do not select from taste, and do not select from what another implementation happens to do.**
In order:

1. **Theorem fidelity** — which mechanism appears consistently in equation (9), in `D_A`, and in
   the appendix-B terms? [The concordance's §7](theorem-concordance.md#7-truncation-is-not-in-the-theorems-algorithm)
   answers it, and A1a promoted that answer from a reading to a stated finding: **one mechanism,
   untruncated, the same in the score and in the influence function.** So this criterion does not
   rank the four variants against a convention the theorem holds — it has none. It asks which of
   them is a finite-sample rendering whose *final* score is the theorem-defined score of the
   estimator being declared, which is criterion 2 made into a design requirement rather than a
   check. Two of the four cannot satisfy it by construction, and saying which is B1b's first
   paragraph. **They are C and B**, and for opposite reasons: C evaluates the residual and the
   denominator at two different mechanisms, so it is no theorem's `D_g` whatever it is the
   first-order condition of; B *is* the theorem's own step, and stays the definition of the
   estimator wherever the bound is slack, but a fitted `g*` is not bounded away from zero the way
   the theorem's `g_0` is assumed to be, so it cannot be the default. A and D survive to criterion
   2, which separates them.
2. **Exact final-score validity** — does the returned state satisfy the exact equation the
   reported correction uses, to the declared statistical tolerance?
3. **Numerical stability** — how does it behave as the clipped share rises?
4. **Substitution-estimator integrity** — is it still a targeted plug-in construction of the kind
   the theorem covers?

There used to be a fifth, *reference fidelity* — whether the variant reproduces `drtmle` where the
two algorithms are meant to agree — ranked below these and never a substitute for 1 and 2. It is
gone with the [retired parity piece](../roadmap.md#closed-since-this-list-opened), and note that it
would have been the weakest criterion here in any case: **the theorem clips nothing at all**, so R's
convention is one candidate among four rather than the reference the others are measured from.

### What B1b reports

For each variant, on each fixed fixture and seed: fraction clipped; each arm's final raw score;
each arm's final reported correction mean; the identity residual; `psi`; `se`; the interval; the
number of rounds; cap/stall status; and the objective value where one exists.

Acceptance: score/correction identity by construction; final reported score below the predeclared
validity threshold; **no silent success when a constrained root does not exist**; and the
concordance row marked *met* or *met under a stated restriction*.

**The fixtures, now that the prototype has said which ones separate anything**, and these are the
ones B1b was accepted on. Three, and each is there for a claim the other two cannot make: `nonlinear` seed 3 — the module fixture, no clipped
rows — for the regression surface, where the selected convention must reproduce today's `psi`,
`se` and stored scores; `nonlinear` seed 2 — one clipped row of 600 — for the ordinary-draw
identity, since one row is enough; and `weak_overlap` seed 0 **at a forced `g_bounds=(0.15,
0.85)`** for the separation, because at the `auto` bound A and D are one fit and the table above
would report a tie. The forced bound is the point rather than a stress setting: it is the only
configuration run so far in which the bound binds *at the fixed point*, which is the only place
the candidates differ.

**And "fraction clipped" stopped being a witness the moment this landed.** It is measured at the
exit, and a convention that carries the bounded array forward exits with `ε → 0` and so with
nothing clipped — 0 on the draw where 375 rows clipped before. Report it still, since zero is what
says the fix took; but a *fixture* has to be selected on something else. This is
[B1a's fifth condition](#1-the-invariants-piece-b1a) needing a new witness rather than a new
intent, and it is [stop-ship 14](../roadmap.md#stop-ship)'s shape arriving in a second place.

**The replacement is not the initial mechanism's clipped share**, which this document proposed and
which is **zero on the draw item 20 was found on**: nothing about that fit's initial mechanism
leaves the bounds, and what clipped was the tilt. It is `CorrectionRow.margin`, how close the
targeted mechanism comes to either bound as a fraction of the interval — `1.2e-06` on that draw
against `0.14` on its sibling, because a constrained root sits *against* the boundary of the
feasible set. Nothing derivable from the returned arrays proves the constraint was active, since
the trajectory is not on the record; what this is is a property separating the two draws by five
orders that the fix cannot manufacture.

### Two acceptance criteria that were wrong and are now stated correctly

**Weak overlap is not required to pass.** An earlier draft made "a `weak_overlap_dgp` fit whose
score check now passes" a B1 deliverable. It conflates a software identity with a statistical
verdict, and it prejudges [B2](#4-the-sweep-piece-b2). Weak overlap still moves `1/g` in equation
(8), `Q_r/g` in equation (9), the *target* `g_{r,2}` is fitted to, the ratio `g_{r,2}/g_{r,1}`,
the effective sample size, the variance and the higher-order remainder, and whether a bounded root
exists at all. So the criterion is:

> On the pinned weak-overlap fixture, the stored score, the recomputed correction score and the
> influence-curve mean agree exactly under the selected convention. Whether the resulting score
> then passes the statistical validity threshold is an outcome for B2, not a condition imposed
> here.

**`psi` invariance is a tolerance, not an axiom.** It is true that the corrections are mean-zero
contributions to the *curve* if B1b changes only how the correction is evaluated after targeting.
It is **not** true if the chosen convention changes the mechanism carried into later outcome
targeting steps, later reduced-regression refits, the closing pass, or the final targeted outcome
regression — in which case the plug-in moves and so does the point estimate. So predeclare

```text
|psi_new − psi_old| ≤ c · se_old        with a small c, or compared against n^(−1/2)
```

and compare each candidate against the estimate produced by **the exact targeted state it claims
to implement**, rather than treating today's `psi` as ground truth. A material movement triggers
investigation; it does not automatically reject the candidate, and it may well reveal that the old
estimate came from a different targeting path.

## 3. The component checklist (piece A1a)

**This was `drtmle` parity and it is [retired](../roadmap.md#closed-since-this-list-opened).** There
is no R here and none in CI. What survives is the decomposition, because the decomposition was
never really about R: only the thing each component was compared *against* was.

**Compare components, not `psi` and `se`.** Several differences cancel at `psi` — a sign error in
one correction, a scaling, a swapped `gr1`/`gr2`, targeting at the wrong starred arrays — and
`psi` is precisely the quantity all three empirical means being zero makes *insensitive* to the
corrections. The list, in the order a discrepancy must be localised: initial `Q̄` and `g`
predictions; each reduced regression; each targeting coefficient; `D*`, `D*_Q` and `D*_g`
separately; the full corrected curve; the three empirical scores; then `psi` and `se`.

Each is checked against **what the derivation gives for it**, on a law whose truth is known. Start
from user-supplied nuisance arrays or a deterministic GLM, never a Super Learner, so that a
discrepancy is arithmetic rather than a fold draw. **Localise to the earliest component that
differs** before reading anything into the ones after it.

### The mapping a component is checked against

| paper object | Python |
| --- | --- |
| `Q̄_n` | initial outcome predictions |
| `g_n` | initial propensity predictions |
| `Q̄_{r,n}` | `ReducedSet.qr` |
| `g_{r,n,1}` | `ReducedSet.gr1` |
| `g_{r,n,2}` | `ReducedSet.gr2` |
| equation (8) | the ordinary outcome score |
| equation (9) | the mechanism score |
| equation (10-uni) | the reduced-outcome score |
| `ψ̂(a)` | targeted counterfactual mean |
| the covariance | rowwise curve covariance |

R's names for the same objects — including the `gr1`/`gr2` inversion, which is the single easiest
thing here to transcribe backwards — are provenance and live in one place, [the concordance's
§13](theorem-concordance.md#13-the-object-concordance), whose `evidence` column is where
*which test pins which row* is recorded. Two copies of a mapping table is one copy too many.

### Four laws, and the shape of each is a decision

Three of the four were already laws rather than exports, which is part of why the retirement costs
so little. **Every one of them must be deliberately misspecified somewhere**, and that is not a
stylistic preference: at the truth `Q_r` and `g_{r,2}` vanish row by row, so a broken
implementation agrees with plain `TMLE` and with any reference alike. That is
[lesson 2](investigation-log.md#what-the-sizings-got-wrong), and it is the reason an
exact-law check is *not* automatically stronger than the parity check it replaced.

1. **Finite support, deliberately misspecified.** A small discrete `W` with repeated nuisance
   values, so the reduced regressions genuinely pool cells and a longhand calculation is possible.
   This is the one that validates definitions and signs without a learner in the way, and it is
   the law [item 21](theorem-concordance.md#4-the-sign-discrepancy-item-21--resolved)'s
   hand-calculation extends: it must carry a **nonzero `Q_r`** and be built so the two candidate
   signs give materially different variances. `tests/unit/test_remainder_drtmle.py` and
   `tests/unit/test_theorem_drtmle.py` are this law already.
2. **Outcome nuisance close but not exact, mechanism wrong.** Deterministic arrays or a
   deterministic GLM. "Close but not exact" is the whole content.
3. **The mirror**: mechanism close, outcome wrong.
4. **The known-uncentred split**, committed as data rather than as a seed — the fold assignment
   and the nuisance outputs, not a call to a random generator whose implementation may move.

Tolerances per component rather than one blanket number: machine precision for the hand-checked
finite-support quantities, `1e-8`-ish for deterministic GLM predictions and coefficients, and
**row-by-row** comparison for the curves rather than a comparison of their variances.

### What is left, and it is a list of tests rather than a fixture to import

**Nothing, and that is A1a's deliverable.** `TODO` is gone from [the concordance's object
table](theorem-concordance.md#13-the-object-concordance)'s `evidence` column; the two rows
still open name their owner — item 13 is piece C's and cross-fitting is A1b's.

The one gap the retired piece had correctly identified and that nothing else covered was **the
reported curve's own decomposition, pinned against a perturbation of the law**. It is
`tests/unit/test_influence_gateaux_drtmle.py`, and the shape it took is worth recording because it
is not the shape planned. The plan was a further oracle carrying the whole DRTMLE limit as an
analytic functional. It needs none: **in the union model at saturated reductions the corrected
curve collapses onto the efficient influence function**, `1/g_1 − g_{r,2}/g_{r,1} = 1/g_0` on the
mechanism side and a cancellation of `Q̄*` against `Q_r = Q̄_0 − Q̄*` on the outcome side. So the
derivative already in the repository is the right-hand side and only the left had to be written.
Two tiers close at `~1e-15` — this module's longhand, and a real `DRTMLE` fit, which is also the
first fit here against a deliberately misspecified law.

Its **blind spots are named in its own docstring**, each measured by running the mutation and
watching it *pass*: item 23, equation (9)'s covariate sign, and a reduced regression's pooling
weight — one degeneracy each time, a cell being blind to a term it sets to zero. That list is the
deliverable as much as the assertions are, for the reason
[lesson 9](investigation-log.md#what-the-sizings-got-wrong) gives.

## 4. The sweep (piece B2)

One dispatch of `benchmarks/bench_drtmle.py`, **after B1**, because every conclusion it could draw
today is read through a curve that a share of fits have wrong.

**The piece split into B2a and B2b, and this section is what both were executed from.** The
dispatch this section describes could not happen until the script recorded what the section asks
for, and the paper's update order did not exist to be run at all — so the instrument was one pull
request ([B2a](../roadmap.md#b2a--the-sweep-instrument)) and the dispatch and its reading were the
next ([B2b](../roadmap.md#b2b--the-dispatch-and-what-it-decides)). That is the same split B1 took and
for the same reason: one half precedes the other and depends on it. **Both have landed**, and the
dispatch was four runs rather than the one this section planned: the main sweep, one order run per
process, and the order arm again at three times the seeds. [What they
measured](investigation-log.md#what-the-b2b-dispatch-measured).

**One instruction here could not be executed as written, and the correction is B2a's.** This
section asks whether the failures persist "when the reductions are handed the **oracle** values",
on the grounds that it "costs nothing because the datasets already know their truth". The datasets
know `Q̄₀` and `g₀`; they do not know the reductions. A reduction is a conditional expectation
given a **fitted** object — `Q_r(a, W) = E[Q̄₀ − Q̄* | ĝ(a|W)]` and the two `g_r` given `Q̄̂(a, W)` —
so its truth is a property of the estimator's own arrays and not of the process, and no closed form
or fresh draw from the DGP supplies it. On these processes the arm is therefore
`--reduced-learner`, which varies the reduction's learner and sees whether the failures move; it is
labelled as a proxy rather than as an oracle. A genuine one here needs the fitted learners exposed
for out-of-sample prediction — `cross_fit_predictions` discards every per-fold model — and is a
construction with its own derivation attached (*which* fold's model is `ĝ` off-sample), which is
**item 24**.

**Where the oracle *does* exist it is now built, and it answered more than the question.** On the
exact law the conditioning variables take three values and the conditional expectations are finite
sums, so `tests/unit/test_oracle_reductions.py` injects them through
`ReductionSpec.refit` — recomputed at the current targeted pair every round, as the real ones are —
and runs a real alternation on them. Three findings, each with its measurement:

- **with the reductions exactly right, the fit recovers the truth while *both* primary nuisances
  are wrong on purpose**: `0.66`, `0.38` and `0.28` to `3.6e-08`, and to `1e-12` under
  `guard=("g",)` where no mechanism equation is solved — which locates that residual at
  [limitation 5](../roadmap.md#limitations-recorded-rather-than-fixed) rather than at the oracle. It is
  `test_remainder_drtmle.py`'s expansion arriving at the other end of the estimator;
- **the saturated learner reproduces the oracle to `1e-14`**, over a whole alternation rather than
  one call, which is the control that says the injection computes the reduction and not something
  else;
- **a wrong reduction moves `psi` by 0.36 to 0.80 of a standard error and leaves every score
  solved.** That is the discrimination this arm was wanted for, and it points the opposite way from
  the phrasing above: a sweep fit whose **scores** fail is not a fit whose reductions were noisy,
  because a bad reduction does not show up there at all. What it damages is the estimate, silently.

**The diagnosis stays widened even though the cause is found.** `1/g` in equation (8) is one of
*five* places weak overlap enters, and B1 accounts for the score failure without saying the other
four are harmless: equation (9)'s covariate is `Q_r/g`; `g_{r,2}`'s own *target* is a quotient by
`g`, formed once at fit time ([limitation 9](../roadmap.md#limitations-recorded-rather-than-fixed));
the ratio `g_{r,2}/g_{r,1}` is unstable when either the numerator is noisy or the denominator
small; and truncating `g` moves not just the covariates but the reduced regressions' estimands,
since two of the three condition on `ĝ`.

What the sweep records per fit, beyond what it records now: quantiles of raw and truncated `g`;
**the share of rows the truncation binds on**, which is now the first column to read; per-arm
effective `n`; the high quantiles of every clever covariate; the distributions of `Q_r`,
`g_{r,1}`, `g_{r,2}` and their ratio; the share of each score carried by the top 1%, 5% and 10% of
rows; the Hessian condition numbers; the scores either side of truncation; `psi` and `se` across a
truncation grid; the identity residual and `B_clip` from [B1a](#1-the-invariants-piece-b1a); and
whether the failures persist when the reductions are fitted by a different learner — the
substitute for the oracle run this section originally asked for, above.

**All of that is in the script as of B2a**, in three tables — *How the alternation exited*, which
is what the first sweep printed; *Where weak overlap enters*, one column per place; and *What the
reported curve rests on*, which is B1a's identity, the standardised scores and the concentration.
Two of the columns above had to change their definition on contact with the code, and both changes
are the same lesson B1b learned about `clipped`:

- **the clipped-row share is read at the *initial* mechanism**, not at the targeted one. Since B1b
  the alternation carries the truncated tilt forward, so a converged fit clips nothing at the exit
  however hard the draw was — a column read there would be zero on every row of the table, which
  is [stop-ship 14](../roadmap.md#stop-ship)'s shape. `margin` sits beside it as the witness that the
  bound had something to do;
- **equation (9)'s Hessian condition number does not exist to be reported.** The bounded solve is
  a root find rather than a Newton step ([B1b](../roadmap.md#what-b1b-landed)), so there is no Hessian
  at that call site. The outcome fluctuation's is reported and describes the closing pass's *joint*
  solve over (8) and (10); `ill` carries equation (10)'s conditioning, as it always did.

A valid **truncation curve** for `DRTMLE` has to refit any reduced regression whose target moves
with the bound. One that moves the denominators and holds the quotient regression's target fixed
is *partial* and must be labelled so.

### The update-order rule, frozen before the dispatch

Item 22's numerical half asks whether the two routes reach the same fixed point. This is the rule
it is judged by, and it is written here before the sweep runs for the reason [§5's
rules](#the-decision-rules-frozen-before-the-dispatch) are: **it may be changed before the dispatch
with a written reason, and not after it.**

The theory that makes it falsifiable: both routes drive the same three empirical means to zero, so
if both are asymptotically linear with the same influence curve then
`ψ_paper − ψ_cleverly = o_p(n^(−1/2))` and `|Δψ|/se → 0`. That is a **claim with a direction**, not
a reassurance — and the direction is the whole of what a single cell cannot show.

> **The two update orders reach the same fixed point if:**
>
> 1. the median `|Δψ|/se` **decreases across the three sizes** in both processes;
> 2. the **count** of draws in which the route difference exceeds the *reseed* difference is
>    compatible with half the pairs at the largest size — the routes moving `psi` no further than a
>    different fold split of one route does;
> 3. the median `se` ratio is inside `[0.95, 1.05]` at the largest size in both processes;
> 4. no fit in either arm fails its score check or its state identity.
>
> The **primary** evidence is clause 1's slope, not any single cell: two sizes are suggestive and
> three carry a rate, which is why this arm runs at three. Clause 2 is the one that can fail while
> every other passes, and it is the clause the control arm exists for.

**What would falsify it**, stated so that the answer is not chosen after the numbers: a route
difference that **does not shrink** while the reseed difference does. That combination says the two
routes are converging to different limits, and item 22's theoretical half — that the paper
prescribes a fixed point rather than a route — would not survive it. A route difference that is
large but shrinks at the same rate as the reseed's is the *opposite* finding, and is the expected
one.

**Clause 4 needed a column, and adding one is not changing the rule.** The identity lives on *What
the reported curve rests on*, and that table is base-only — so at the time this rule was frozen
three of its four clauses were answerable from the printed tables and the fourth was answerable for
one arm of the two it names. Every `Exit` already carried the number, since `one_fit` computes the
whole `Curve` whatever the arm; only `comparison_rows` dropped it. It now carries a `worst identity`
column, a **max** over the cell as `curve_rows` takes it, since an identity's right value is zero and
a median would let one broken fit hide behind eleven sound ones. The distinction this paragraph
exists to hold is that the *rule* is unchanged and what moved is what can be read against it —
`tests/unit/test_bench_drtmle.py` pins the column, watched to fail. A clause nobody can evaluate is
not a frozen rule, it is a frozen intention.

**Why a count rather than an interval.** `comparison_rows` reports a median, and this repository has
no Monte Carlo standard error for a median — `EstimandSummary.bias_se` is a mean's. At twelve draws
a distribution-free paired count is honest where an invented interval would not be, so clause 2 is
stated on the count and the mean with `sd/√M` is reported beside it for continuity. Raising `--seeds`
for this arm is the way to sharpen it; reading its median as though it were a coverage number is not.

**That sharpener was taken and the rule still did not resolve, which is worth recording against the
rule rather than against the estimator.** [B2b](../roadmap.md#b2b--the-dispatch-and-what-it-decides)
ran the arm at twelve draws and again at thirty-six. Clause 2 fails on both processes at both
counts, always *short* of half. Clause 1 is met on one process at each count — a different one each
time — because a median over twelve or thirty-six draws is not stable enough to carry the claim; on
`nonlinear` at 36 draws it rises by 2% in the last step while the control's median rises in the same
place. What *is* stable is the route difference sitting 3.5 to 4 times below the control's at every
cell. **Two restatements have reasons behind them and may be made before a further dispatch, never
after one**: clause 2 should be **one-sided**, since a count far below half is evidence for the
conclusion rather than against it; and clause 1 should be stated on the **ratio of the two medians**
rather than on the route's alone, since the control exists precisely to absorb what a refit does and
the route median inherits its noise otherwise. Neither is made here. [The
reading](investigation-log.md#the-same-rule-at-thirty-six-draws-and-why-the-two-readings-are-not-nested)
carries both seed counts.

**And the two readings are not nested, which is a property of the instrument.**
`bench_drtmle.py` slices its three seed streams as `[:s]`, `[s:2s]` and `[2s:]`, so raising `--seeds`
holds the *data* seeds' prefix and moves the fold and control blocks wholesale: the 36-draw run
shares its first twelve datasets with the 12-draw one and none of their fold splits. Neither
supersedes the other and both are kept. The script's comment now says so.

**The dispatch it is judged from** is its own, because three sizes across four processes and three
arms does not fit the runner:

```text
processes: nonlinear weak-overlap    sizes: 600 1200 2400    seeds: 12
order: true    order_control: true
```

2 processes × 3 sizes × 12 seeds × 3 arms = 216 fits, ≈ 100 minutes at `jobs: 2` against the
180-minute cap, with the paper arm's longer route allowed for. The main four-process sweep keeps its
two sizes and runs with the arms **off**.

**It was dispatched as two runs, one process each, and the precaution turned out to be
unnecessary.** The reasoning was the cap: the estimate above is against the base arm's 42.6s per
fit, the paper arm took 22 rounds against 8 on the draw the two were first compared on, so a *draw*
costs base + reseed + paper and 72 draws at `jobs: 2` reaches ~118 minutes before the `n = 2,400`
cell is allowed for — and the workflow prints nothing until every fit has returned, so a run killed
at the cap reports no table at all rather than a truncated one. Every table here is keyed on
`(process, n)` and every clause of the rule above is stated per process, so two dispatches produce
exactly the rows one would have. **The split changes the run and not the rule**, and it is recorded
here rather than in the log because a reader checking the rule against the evidence will otherwise
find one dispatch promised and two delivered.

**What the cost model got wrong is worth more than the split.** The two runs took **722s and 393s**
for 108 fits each, at 9.8s and 6.4s a fit — against the 42.6s every sizing on this page was written
from. The main four-process sweep is likewise 378s where it was 2,588s. The cause is
[item 7](../roadmap.md#closed-since-this-list-opened)'s exit criterion: the alternation now reaches
its tolerance in 4 to 9 rounds where it stalled after 12 to 24, so the sweep does about a seventh of
the work. **Every runtime estimate in §4 and §5 is therefore stale in the same direction**, and the
consequence for [§5's study](#5-the-controlled-study-piece-c) is the significant one: it was sized
at "~24 hours serial and about two on a 12-way `matrix:`" from a 43s `DRTMLE` fit at `n = 1,200`,
and that fit is now several times cheaper. Re-time before re-sizing rather than dividing by seven —
piece C fits both estimators over every replicate at three sizes, and only the `DRTMLE` half moved.

**That re-timing has since been done and the instruction was the right one.** C1 measured the same
fit at the same size at **5.6s** — a factor of 7.7, so dividing by seven would have been close — and
a **Tier-1** fit at **1.2s**, which dividing by anything would not have reached, since its primary
nuisances are prescribed functions rather than learner fits. [§5's cost
paragraph](#sizes-and-replications) carries both.

**Stopping and validity are two questions and the sweep must report them separately.**
Asymptotic linearity asks for `P_n D = o_p(n^(−1/2))`; the honest finite-sample rendering of `o`
is a deterministic sequence `c_n/√n` with `c_n → 0` slowly — a **numerical** criterion, stated as
one — with the standardised score `|P_n S_j| / sd̂(S_j)` reported afterwards as a separate
diagnostic rather than folded into the stopping rule. That separates when to stop iterating from
whether the fit that came out is entitled to a Wald interval. `targeting._solved`'s
`_NEGLIGIBLE / n` currently conflates them and assumes `se = O(n^(−1/2))` on the scaled outcome
rather than measuring it; it is conservative exactly where it matters — under weak overlap `se` is
large — but a fit with a very small `se` is the untested direction.

**The product decision belongs here**, and B1 changes what it is likely to be. If the sweep still
finds no stable region, `DRTMLE` should refuse or invalidate under weak overlap on a
**predeclared** diagnostic rather than warn. But the evidence that motivated the refusal was 23 of
24 failed score checks, and on present measurement those are the convention mismatch rather than
the estimator breaking down — so **do not predeclare the refusal before B2 re-measures**. What
survives regardless is the ordinary positivity warning, which fires on these fits already (29% of
units outside the bounds on the seed-0 draw).

## 5. The controlled study (piece C)

**This section is the specification and [the coverage study](coverage-study.md) is the design.** What
follows is what a study has to contain to be believed; what the cells actually are, which constants
were committed, and what the instrument has measured are there. The pair is deliberate — a rule
restated next to the numbers it judges is a rule that can differ from itself.

**C is three pull requests and two have landed.** **C1** is the harness and Tier 1 complete;
**C2** is Tier 2's prescribed-rate learners plus the fold-retained nuisances `P₀D̂` needs, and so
item 13; **C3** is the pilot, the freeze and the study, and so item 3. The split is this page's own
grouping rule — shared *evidence*, not shared subject matter — and the two tiers share none: Tier 1's
remainder is a **quadrature**, since both its nuisances are prescribed functions of `W`, so
`n^α R₂ → c` is an identity a unit test asserts; Tier 2's needs the values of *fitted* reduced
regressions on covariates no fold trained at. So **the tier that reads item 13's rate off is C2**,
and Tier 1's exact remainder is the regime-entry column beside it rather than a substitute for it.

The regime wanted is one where `R₂ = ‖ĝ − g₀‖·‖Q̄̂ − Q̄₀‖` does not vanish fast enough for a plain
`TMLE` interval while one nuisance is still consistent. Four things the study has to contain, each
ruling out a way of being believed for the wrong reason: **both off-diagonal cells**, a **genuinely
slow nuisance**, **coverage against its Monte Carlo standard error**, and **a size trend** — three
sizes if the budget reaches, since two are suggestive and three carry a rate.

One trap, already met once: `tests/e2e/test_double_robustness.py`'s "correct" cell is an **oracle**
(`OracleOutcomeContinuous`, `OracleTreatment`), which makes the good nuisance exactly right, `R₂`
exactly zero and `TMLE`'s interval already valid. The gap opens only where the good nuisance is
*estimated*.

### Two tiers, because they answer different questions

- **Tier 1, prescribed sequences.** A test-only nuisance-injection interface handing the estimator
  `Q̂ = Q̄₀ + n^(−α)·h_Q` while `ĝ → g₁ ≠ g₀`, and the mirror. No learner, no fold draw, a bounded
  perturbation keeping every probability interior. This is not an applied claim and must not be
  presented as one; it is the only construction in which "the intended asymptotic regime was
  entered" is true *by definition*, which makes it the right place to read item 13's remainder
  off.
- **Tier 2, prescribed-rate learners.** A series, spline or histogram regression with a smoothing
  sequence chosen in advance, so the rate is analysable and reproducible. **This is the
  demonstration.** A Super Learner's realised rate is neither identified nor reproducible, so a gap
  it produces could as easily be finite-sample instability as the intended drift; keep it for the
  applied stress tests that come after.

  **Landed with C2**, as an oversmoothed **additive kernel** with `h_n = 1.15·n^(−0.125)` rather
  than as a regressogram, and the substitution is a finding rather than a liberty: a regressogram's
  bias oscillates within every bin, so its `L₂` norm is `O(B⁻¹)` while its *inner product with a
  smooth weight* is `O(B⁻²)` — and the remainder is an inner product. Matching a declared remainder
  rate with one needs a bin count at which the fit is variance-dominated at these sizes, and the
  remainder it then produces is sampling noise. This list is illustrative; what it asks for is a
  sequence chosen in advance, and a bandwidth is one.
  [The design note](coverage-study.md#tier-2-a-prescribed-rate-rather-than-a-prescribed-sequence)
  carries the constants.

### The drift coefficient, which a rate alone does not give you

For one arm, with `Q̂_a − Q_{0,a} = n^(−α)h_a + o(n^(−α))` and `ĝ_a → g_{1,a} ≠ g_{0,a}`,

```text
R_{2,a} = n^(−α)·c_a + o(n^(−α)),      c_a = P_0[ (g_{1,a} − g_{0,a})/g_{1,a} · h_a ]
√n·R_{2,a} = n^(1/2−α)·c_a + o(n^(1/2−α))
```

so `α < 1/2` gives growing root-`n` drift, `α = 1/2` a fixed local shift, and `α > 1/2` none —
**provided `c_a ≠ 0`.** That proviso is the whole correction: the remainder is an **inner
product**, not a norm, so `c_a` can vanish because `h_a` is orthogonal to the misspecification
weight even though `‖h_a‖ > 0`, and `c_1 − c_0` can vanish in the ATE while both arm coefficients
are nonzero. A finite-support example with `c_1 = c_0 = 0.2` has first-order drift in both
treatment-specific means and **exact cancellation** in the ATE.

So `α < 1/4` is the familiar bar for the *both-consistent* product condition and is sufficient
rather than necessary here — and `α < 1/2` on its own is not sufficient either. The design must:

- choose `h_a` **analytically** so `|c_a| ≥ c_min > 0` for each arm evaluated and
  `|c_1 − c_0| ≥ c_min > 0` for the ATE;
- **commit the coefficient calculation with the design**, before any fit is run;
- verify empirically that `n^α·R_{2,a} → c_a` and `n^α·R_{2,ATE} → c_1 − c_0`.

**And `c_a` is not the whole of it, which is C3b's correction to this list.** The three bullets
above are conditions on the *plug-in* remainder, and a fit's bias is the same inner product at the
**targeted** regression. Targeting solves `P_0[w_a(Q̄*_a − Q̄_{0,a})] = 0` with `w_a = g_{0,a}/ĝ_a`,
and the remainder's weight is `u_a = 1 − w_a` — so the score removes precisely the component of
`h_a` the fluctuation can reach, and a design satisfying the bullets above has constrained
**nothing** about what survives it. Measured at C3's own Tier-1 design, `c_ATE` was `0.40` by
construction and the surviving coefficient was `0.00092`.

The repair is a *second* coefficient rather than a larger first one, and it is available on the
same terms: eliminating `ε_a` through the score leaves

```text
b_a = P_0[ v_a · h_a ],      v_a = 1 − kappa_a w_a,     kappa_a = P_0[S_a] / P_0[w_a S_a]
```

with `S_a` the direction the fluctuation's one free parameter per arm moves `Q̄_a` in. So `b_a` is
a linear functional of the shape exactly as `c_a` is, and a design can be *built* to hit it — a
2×2 solve in the span of the two representers, not a projection and a renormalisation. **A fourth
bullet, therefore: choose `h_a` so `|b_a|` and `|b_1 − b_0|` are bounded below too, declare both
coefficients, and read the regime off the second.**

Two consequences a design has to expect. **Declaring `b = c` makes the injection orthogonal to the
score**, so `ε → 0` and the two columns coincide — a useful special case rather than a coincidence.
And **the arms' opposite signs do not carry over**: `c_ATE` a sum of magnitudes says nothing about
`b_ATE`, which at C3's Tier-1 design was a *difference* because both arm coefficients came out
positive. Declare the sign structure on `b`, since that is the contrast the estimand has.

Nuisance-error norms alone do not demonstrate that the intended drift was entered. What argues
*against* pushing `α` very small is the other side of the ledger: the appendix-B terms `DRTMLE`
needs to be negligible involve the reduced regressions, whose targets are built out of `Q̂` and
`ĝ`, so a badly enough estimated primary nuisance degrades the corrected estimator too. Choose `α`
so the `TMLE` gap is visible at reachable sizes, state it in the design note, and treat a `DRTMLE`
failure at small `α` as a finding about the appendix-B conditions rather than as a bad setting.

### Verifying the regime was entered

Per size, against the truth the DGP knows: `‖Q̄̂ − Q̄₀‖` and `‖ĝ − g₀‖` with their log-log slopes
and uncertainty across replications; the misspecified nuisance's distance to *its own* limit
staying bounded away from zero; positivity stable across sizes; `√n·R₂` failing to vanish for
`TMLE` while `√n·R_remaining` does vanish for `DRTMLE`; and the realised drift coefficients above.
Without these columns a correct coverage number is still only a number.

**And the coefficient has to be verified at the *targeted* regression, not at the initial one.
This clause is new, it is the second change these rules have taken, and the written reason is that
[C3's pilot](coverage-study.md#what-the-pilot-measured) failed on exactly its absence.** The
remainder a design commits a coefficient for is

```text
R_2(Q-hat)  = P_0[ (ĝ − g_0)/ĝ · (Q̂ − Q̄_0) ]        the plug-in remainder
R_2(Qbar*)  = P_0[ (ĝ − g_0)/ĝ · (Q̄* − Q̄_0) ]       the estimator's bias
```

and **a design can hit the first exactly while the second is twenty times smaller**, because the
fluctuation's score equation constrains a weighted offset of `Q̄* − Q̄₀` and the plug-in quantity is
not subject to it. Measured: `n^α R₂(Q̂)` at `+0.4000` at every size, against a `√n` bias of `0.1`
to `0.6` where the sizing predicted `2.5` to `4.2`. So a study that verifies only the plug-in
coefficient has verified that its *injection* is what it says and **not** that the regime was
entered — which is the one thing this section exists to establish.

Operationally: report both columns, and read the *targeted* one against **its own** declared
coefficient — `n^(−α)b`, not `n^(−α)c`, which is the further correction
[the coefficient section](#the-drift-coefficient-which-a-rate-alone-does-not-give-you) makes.
`benchmarks/drtmle_tier1_bias.py` computes them side by side on the same rows of the same fits and
runs in seconds a size, so this is a pre-flight check rather than a study — and it must pass
**before** a coverage dispatch, not be inferred from one afterwards. The same discipline the
invalid-fit rule already has: written down before the numbers exist.

**C3b landed the instrument that reads it**, and read it: the harness prints a *"which regime the
fits entered"* table beside the design's predictions, on **both** estimators — the plain `TMLE`'s
row is the one this clause is about, since that is the estimator whose interval a shortfall is
claimed against — and a **pre-flight table** carrying all three conditions as a verdict. The three
are stated in [the design note's repair section](coverage-study.md#the-repair-and-what-would-say-each-half-of-it-is-wrong)
and are not restated here, for this document's standing reason.

### Evaluating `P₀D̂`, which is not automatic for a cross-fitted fit

**Never substitute `P_nD̂`** — that is the quantity targeting drove to zero, so it answers a
different question. But out-of-fold prediction arrays at the observed rows do not define functions
on new `W`, so integrating `P₀D̂` needs the fold-specific primary nuisance models, the
fold-specific reduced-regression models, the fold-specific targeted transformations, and a stated
convention for averaging or conditioning over folds.

- **Tier 1**: exact finite-support summation or a very large independent draw. No model retention
  is needed, because the nuisance sequence is prescribed.

  **As built, that holds of the *plug-in* remainder and not of the corrected one, and the
  distinction is what put the two tiers in different pull requests.** `R₂` at the injected sequence
  is a Sobol quadrature over two prescribed functions and is exact — that is C1's regime-entry
  column, closing on the declared coefficient to five figures. `R_remaining` is not: the three
  *reduced* regressions are fitted whatever the primary nuisances are, so `P₀D̂` needs their values
  off the training rows and the retention below is required in Tier 1 too. C1 therefore reports no
  corrected remainder and prints no column for one.
- **Tier 2**: add a **benchmark-only** fitted nuisance object exposing `predict(new_data)` per
  fold; evaluate each fold's corrected curve on an independent draw using the nuisance functions
  trained for that fold; average the fold-conditional `P₀` values with the same fold weights the
  estimator uses. A completely independent training/evaluation split is the alternative. **This is
  C2's and it carries item 13 with it**, for the reason just given: it is what the corrected
  remainder needs at *either* tier.

  **C2 landed it as a library keyword rather than as a benchmark-only object**, and the departure
  from this paragraph's own words is worth the sentence. Retaining the models and *replaying* the
  alternation outside the library is a second implementation of `solve_with_reduction`'s state map
  — and that map is the hard part: the outcome solve applies its tilt once per Newton step and
  shrinks after each, `solve_bounded_mechanism` clips, and the reductions are refit every round, so
  `Q̄*` is not `expit(logit Q̄̂ + ε·H)`. A bug in a replay is indistinguishable from a real
  remainder. So `DRTMLE(evaluation=…)` carries the evaluation rows **through the same solvers**, as
  `Fluctuation.carried` already does for the nested construction, generalised in one way: a carried
  item supplies its own clever covariate, because the evaluation rows' is not the fitting rows'.
  It is inert when absent — pinned bit for bit — and it is anchored by an identity rather than by
  an argument: handed the fitting frame back as its own companion, fold `k`'s slab must reproduce
  the production array at the rows fold `k` holds out (`tests/unit/test_drtmle_companion.py`).

**Document the conditioning convention.** Without it `R_remaining` can be an artefact of how
fold-specific fits were extrapolated to the integration sample rather than a property of the
estimator.

**The convention in force is the fold-weighted one**, which is what this section asks for and what
C2 implements: `P₀D̂ = Σ_k (n_k/n)·E₀[D̂^(k)(O)]`, with `n_k` the rows fold `k` holds out and the
expectation over an independent draw whose size is `--evaluation-n`. That size is a **quadrature
rule and not a sample size**: it appears in no root-`n` scaling, which is taken at the fitting
size. Its error lands directly in a replicate's `R_remaining` at `O(m^(−1/2))`, so the harness
draws an independent evaluation sample per replicate and reports the Monte Carlo standard error of
the mean beside every entry.

**E1 takes that sentence at its word, and the rule moves — before the final run and with the reason
written down, which is what the freeze rule below permits.** The fold convention is unchanged; what
changes is what `E₀` is taken by. `--evaluation-n` said *quadrature rule* and was an i.i.d. draw,
which is the one kind of quadrature whose error does not fall faster than `m^(−1/2)`; the
deterministic rule is that sentence's conclusion rather than a departure from it. Four things, and
the third is the one that is easy to skip.

*The reduction, which is why this is not simply a bigger draw.* The corrected curve is **affine in
`Y` given `(A, W)`** and reads `A` **only through the indicator**: `D*_g = (Q_r/g)(1_a − g)` has no
`Y` in it and `D*_Q = 1_a·(g_{r,2}/g_{r,1})(Y − Q̄*)` is affine in it, so

```text
P₀D̂  =  E_W[ Σ_{a ∈ {0,1}} g₀(a|W) · D̂(W, a, Q̄₀(a, W)) ]
```

is an **identity**. Two of the three coordinates integrate in closed form and a quadrature is left
in `W` alone. `--quadrature-points` is the lever: the companion is every Sobol point of the law's
own rule at both arms, `Y` at `Q̄₀(a, W)`, weighted by `g₀(a|W)`.

*`ψ₀` moves onto the companion's grid, and that is most of what the rule buys.* Substituting the
curve's centring, `R_remaining = E₀[D̂ᵘ − Q̄₀] − PₙD̂`, and the equality holds **only if both
expectations are one integral** — under which every term of the integrand is a product of two
nuisance errors rather than an `O(1)` quantity, so the grid's relative error acts on something of
order `n^(−α)`. Taking `ψ₀` from the finer default rule instead differences two grids and the
`O(1)` part survives. `benchmarks/drtmle_remainder.truth_at` is that, and it is why `DGP.quadrature`
exists.

*The error changes kind, and E1 got the second half of that wrong — so the rule moves once more.*
The draw's error is **noise**: independent per replicate, so a study averages it down and it
inflates the spread each Monte Carlo error is computed from. A grid at **one fixed scramble** is a
**bias** — the same points at every replicate — so no replicate count removes it, and E1 shipped a
nested convergence ladder as the thing that would bound it. A successive difference between two
rungs bounds nothing without a convergence result the Tier-2 integrand does not have, so that
paragraph asserted what it claimed to measure.

**E1b's rule is therefore an independent scramble per replicate**, seeded from a stream disjoint
from the data and fold streams, and it is a strictly better trade than the fixed grid rather than a
retreat from it:

- a randomised quasi-Monte Carlo rule is **unbiased at every point count** — the randomisation is
  over the scramble and not over the points — so `E₀`'s estimate is mean-zero-error again and a
  study averages the grid's error down exactly as it averaged the draw's, while being several-fold
  smaller;
- the error becomes **estimable by replication rather than by refinement**: independent scrambles
  at one grid give a standard error assuming no rate, and `benchmarks/drtmle_companion_grid.py` is
  where that is measured, conditionally on each fit;
- and the same randomisation is what makes the *attribution* identified, since a fixed grid's error
  is a deterministic function of the fitted curve and can covary with the remainder, where a
  randomised one cannot.

**The same scramble must serve `quadrature_frame` and `truth_at` within a replicate.** The
cancellation two paragraphs up is the whole reason `ψ₀` moves onto the companion's grid, and it
holds only if both expectations are one integral — which now means one *randomisation* as well as
one point count. A replicate that drew its grid from one scramble and its truth from another would
difference two rules and put the `O(1)` part of the error back.

**Nesting is preserved within a scramble**, so a ladder is still a `Window` on one fit; what the
ladder now reports is stability, and the error column beside it is the across-scramble spread.

*Both rules stay, and passing both is refused.* The i.i.d. draw is the independent route the
deterministic one is checked against — two renderings of one population integral, which is the
strongest check this section has and the same argument `plain_remainder` is checked against Tier 1's
quadrature under. It also remains the default, so every invocation the
[study manifest](study-manifest.md) records reproduces bit for bit.

**What it does not fix, stated because the temptation runs both ways.** The rule's error is *part*
of a replicate's spread and not the whole of it; the rest is the estimator's own second-order
sampling variation, which only a replicate count reduces. Removing the quadrature narrows the bar
condition 3 is read against by whatever share the ladder measures, and the honest reading of a
column still flat afterwards is that the flatness is the estimator's. **Whether it will still be
flat is not a question this rule change answers**, and neither the claim that it will nor the claim
that it will not belongs to a pull request whose subject is the instrument — both are readings of
the rate, and the rate is read once, at the final dispatch, against clauses frozen before it.

### Reporting `R_Q` and `R_g` separately

The single `R_remaining = ψ̂ − ψ_0 − (P_n − P_0)D̂_DR` is necessary and not sufficient: a total
trending to zero can conceal cancellation between the two appendix branches. Where the DGP
permits, report `R_Q` and `R_g` separately, their component products, their signs, and the total.
See [the concordance's §5](theorem-concordance.md#5-the-remaining-remainder-terms) for the
exact terms.

**What C2 reports, and what it refuses, written down before the numbers exist.** Two observations
make the branches computable and one keeps the rest honest.

The branch **sums** need fewer limits than the terms do: writing out `R₃ + R₄` and `R̃₅ + R̃₆`, the
univariate limits `Q̄_{0,r}`, `g_{1,0,r}` and `g_{2,0,r}` **cancel**, leaving the fitted reductions
— which the companion has exactly — and the two `0n` limits. A `0n` limit is a population
conditional mean of a computable quantity given two computable scalars, so it needs no *modelling*
choice: estimated by a binned average over the evaluation draw, at two bin counts, with the
difference reported beside the column. **A branch smaller than that difference is reported as not
having settled** rather than as a number — this section's *"where the DGP permits"*, said out loud
rather than discovered after a dispatch.

> **This paragraph called that difference the limit's "own error" and called the estimate a
> "quadrature and not a fit", and both are withdrawn.** A binned average over a continuous design
> *is* a fit — a regressogram — and its error is a smoothing bias, not a quadrature error. And the
> movement between two bin counts is a **successive difference between two rungs of a refinement**,
> which is precisely the statistic
> [E1b withdrew](../roadmap.md#what-e1-landed-and-what-e1b-withdrew) for the quadrature ladder:
> it says a sequence settled and not *where* it settled. Measured there, the finest rung's `delta`
> ran four times below the true error and three orders above it two rungs earlier;
> `tests/unit/test_drtmle_remainder_study.py`'s `test_settling_is_not_sufficient` constructs the
> binned case, where the movement is `2e-15` and the residual is the whole of the target.
>
> **One direction of the inference survives, and the suppression uses only that one.** A branch
> moving more than its own magnitude is an instrument visibly still moving and is not a number to
> read. A branch that has settled may still be wrong by any amount, because a smoothing bias can
> be stable across two resolutions and large at both. So settling is **necessary and not
> sufficient**, and a settled branch's error is *unestablished* rather than small.
>
> **Randomisation does not rescue this one**, and that asymmetry is the reason E2 exists rather
> than a second application of E1b. An independent scramble makes a *quadrature* error mean-zero
> and estimable by replication; every randomisation here would share the same bin count and the
> same edges-from-quantiles rule, so the across-scramble spread is orthogonal to the bias in the
> partition. What establishes the error is a reference whose fidelity is measured against
> something other than its own refinement — [E2](../roadmap.md#e-what-c3c-handed-back).

The empirical-process terms `M₁` and `M̃₂` are **refused by name**. They are `(P_n − P_0)` of a
difference of estimated curves, and under the fold convention above `P_n` and `P₀` are taken at
different renderings of the nuisances — out of fold on the fitting sample, fold-conditional on the
evaluation draw — so there is no single-sample expression that is both. Picking one and calling it
the theorem's term would be worse than not reporting it. What is reported is each branch's
**second-order half**, which is the half gate 1's clause 4 is about: an empirical-process term is
`o_p(n^(−1/2))` under the Donsker and `L₂` conditions above and carries no product of nuisance
errors to cancel against.

**The bin counts stay at `(12, 24)` and E1 did not move them, which is a decision rather than an
omission.** C3c's `cancel` reaching `1.99x` and `branches settled` falling to `192/250` are gate
1's clause 4 straining on its second half, and moving the instrument that reads a clause inside the
pull request that measures the instrument's *precision* would leave the two unattributable — the
mistake [lesson 14](investigation-log.md#what-the-sizings-got-wrong) is about, in the other
direction. There is also a coupling that makes a third count worse rather than better at the
present rules: two designs at 24 bins is 576 cells, and at 48 it is 2,304, so a finer grid needs
proportionally more rows before its cell averages mean anything, and a bin count raised without the
rows behind it drives every branch toward its own target and reports a spuriously small one. What
E1 does instead is **report the movement that was already being recorded**: `branch_error` has been on
every replicate since C2 and was read by no table, so `192/250` arrived with nothing beside it.
Both harnesses now print it, and the ladder prints it at every rung — which is where the coupling
above becomes visible rather than argued. A third bin count is E5's pre-flight question, to be
taken **with** the row count that supports it.

**What `192/250` is a count of**, now that the reading above is withdrawn: the replicates whose
branches moved *less* between the two bin counts than the branch's own magnitude. It is a
**stability** count. In 58 of 250 the two counts disagreed by more than the branch itself, which
says the binned limits were still moving there and says nothing about how far either is from the
truth. So clause 4's second half is **unread** rather than failed — and the clause's verdict does
not move, because it fails on its first half alone: `√n R_rem` is flat in `q-drift` and does not
fall in `g-drift`, and that column has a measured replicate spread beside it. *`unresolved` is not
a weak `pass`*, and it is not a weak `fail` either.

**The two candidate replacements, and this section does neither.** A randomised bin *origin* would
make the partition's error mean-zero and estimable by replication — E1b's device one level down,
available because the edges are the only thing that has to move. Or E2's reference reduction, which
is a proper estimator of the conditional expectation with a fidelity gate that is not a refinement
difference. The first is E5's pre-flight question with the third bin count; the second is E2's.

### What to report

Per estimator, per cell, per size, per seed batch: bias, `√n` bias, empirical sd, mean estimated
`se` and their ratio, coverage, interval width, rejection rate under a null variant,
targeting-failure rate, the share of intervals marked invalid, the correction terms' own means,
variances and covariance, both remainder diagnostics and the two branches, the nuisance errors and
their slopes, the realised drift coefficients, elapsed time, and a Monte Carlo standard error
against every one of them.

**And the three truncation witnesses**, which are gate 1's new clause 0: `clip share` at the initial
mechanism, `margin` at the exit, and `g_{r,1}`'s distance from its bound. They are not diagnostics of
a fit going wrong — a bound-active fit can have every identity at `1e-17` and every score negligible,
and on `weak-overlap` it does. They say which *estimator* the row is evidence about, which is a
different question and one no other column on this list answers.

All three are on the fit since C1 — `CorrectionCheck.contract`, `initial_clip_share`, `margin`,
`gr1_margin` — so the harness reads them rather than recomputing any, and **the label is reported as
a count of the cell's draws rather than as a cell-level verdict.** That is not a presentation choice:
C1's first run put one to two of six *well-overlapped* draws on the far side of the contract, which
means cells are **mixed** and a median of the margins would report a mixed cell as though it were a
pure one. How to read a mixed cell's coverage number is C3's decision, to be taken **before** its
dispatch under the rule that these may be changed before the final run and not after it.

**Three things E1 adds, and each is about telling an instrument's error from an estimator's.**

*The evaluation rule's own error, beside the column it lands in.* `rule err` on the remainder
table and in pre-flight condition 3's reading, so that a column reported as flat can be read
against the precision it was measured at. C3c's `√n R_rem` came with a replicate spread and
nothing else, and the two candidate explanations for `1.427 ± 0.091` — a flat quantity and a
quadrature too coarse to see the decline — could not be separated from that. The verdict rules are
unchanged and still read the replicate spread, which is right, because the rule's error is inside
it; what the column buys is **attribution**.

**E1b changes what that column is.** E1's witness was the movement when half the companion's rows
are dropped — a fair reading of a bias, a `1.4x` overstatement of a noise, and, on a deterministic
grid, not a bound at all. Under the randomised rule the witness is the **across-scramble standard
deviation at a fixed fit**, which needs no model of either failure mode and is the quantity a
standard error is. A share of the column attributed to the rule must be reported **with an
interval**, and never as one minus a ratio of two marginal variances, which identifies that share
only when both rules' errors are mean-zero given the fit.

*Which rule, and how many rows.* `companion_rule` and `companion_rows` on every record. C3c's
artefacts carry neither, so a reader of them has to know the invocation — and once there are two
rules that is a property of the row rather than of the run.

*The full score row per fit, as a second artefact.* `valid` plus the two failure counts stay
exactly as they are, because clauses 2 and 3 read them apart; the rows are **in addition**, keyed
`(cell, n, data_seed, estimator)` and written under the same timestamp so the two files join. A
count cannot say which equation missed, by what ratio against its threshold, or whether it started
large and was driven down — `score_initial`, and it is the field that separates targeting having
worked from targeting having had nothing to do. Nothing is filtered: writing only the failing rows
would lose exactly that. This is what a replay of the invalid fits reads.

### Sizes and replications

At least three sizes — `600 / 1,200 / 2,400` is the shape, adjusted upward if the prescribed rate
is not visible — and a pilot of 50 to 100 replicates per cell before anything is frozen. The
frozen study wants **250 at minimum and 500 if the budget reaches**: at a true 0.95 the Monte
Carlo standard error of a coverage estimate is `0.014` at 250 and `0.010` at 500, so 250 resolves
a `0.95`-against-`0.88` gap comfortably and does not resolve `0.95` against `0.93`. Then an
independent second seed batch, run after the first is complete. Changing sizes or counts *after*
seeing coverage is permitted only as a new experiment, documented as one.

**What it costs, re-timed with C1.** The figure this paragraph carried — 43s per `DRTMLE` fit at
`n = 1,200`, so ~2,000 fits and ~24 hours serial — was measured before piece B1b and before the
exit criterion item 7 replaced. Re-measured on a four-core container: that same fit is **5.6s** — a
factor of 7.7, which is the *"seventh of the wall clock"* B2b's corrected exit criterion bought —
and a **Tier-1** fit is **1.2s**, because its primary nuisances are function evaluations rather
than learner fits and what it pays for is the alternation. So the Tier-1 pilot — 2 cells × 3 sizes
× 50 replicates, 300 draws and 600 fits — is under an hour at `jobs=2`. **Tier 2 was expected not
to be**: its nuisances are fitted, which is what the 43s was measuring, so C2 re-timed before
re-scoping rather than inheriting either number — and measured **5.4s to 7.4s** per fit at
`n = 600` with a 2,000-row companion, so it is affordable too. The additive smoother is cheaper
than the boosting library the 43s was measuring. `.github/workflows/drtmle-coverage.yml` is the dispatch-only
workflow — a `matrix:` over the cells, since both estimators of a pair must be fitted in one worker
for the shortfall to stay paired — and the nightly tier must not absorb it.

### The decision rules, frozen before the dispatch

They may be changed before the final run with a written reason. They may not be changed after it.
And they are **two** rules, not one, because a correctly implemented estimator that attains
nominal coverage while `TMLE` under-covers by only 0.03 is statistically validated and
commercially uninteresting, and those are different conclusions.

**Gate 1 — statistical validity.** `DRTMLE` is theoretically and computationally validated if:

0. **the supported contract is frozen before the dispatch and the study's cells are inside it.**
   This clause is new, it is the one change made to these rules since they were frozen, and the
   written reason the rules require is [item
   25](../roadmap.md#the-supported-contract-and-item-25): the guarantee is claimed for a fit whose
   truncations are inactive, so a study that does not report which side of that line its cells fell
   on cannot say what its coverage number is evidence about. Operationally: report `clip share` at
   the initial mechanism, `CorrectionRow.margin` at the exit and `min gr1` per cell per size; a
   cell with any of the three active is reported as bound-active and read as *empirically
   supported, outside the theorem* rather than as evidence for or against Theorem 1. The rule is
   changed **before** the final run and not after it, which is the only time it may be;
1. theorem concordance closes, including
   [item 21](theorem-concordance.md#4-the-sign-discrepancy-item-21--resolved);
2. zero state-identity failures from [B1a](#1-the-invariants-piece-b1a)'s checks across the whole
   study;
3. every required final score is negligible under the predeclared validity rule;
4. `√n·R_remaining` trends to zero in **both** off-diagonal cells, and neither appendix branch is
   large with the other cancelling it;
5. `DRTMLE`'s `se` ratio is in `[0.90, 1.10]` at the largest size in both cells;
6. `DRTMLE` coverage is compatible with 0.95 at the largest size in both cells, by the rule below;
7. the qualitative conclusion reproduces in the second seed batch.

**Gate 2 — practical release value.** `DRTMLE` is promoted to a supported public feature only if:

1. it yields a predeclared, practically meaningful improvement over `TMLE` — the proposal is
   `TMLE` short by at least **0.05** in at least one cell, with the Monte Carlo interval on the
   *difference* excluding zero, and `√n·R₂` failing to vanish in that cell;
2. the invalid-fit rate is below its predeclared threshold in the well-overlapped cells;
3. the computational cost is acceptable;
4. the advantage persists in at least one applied stress setting.

The 0.05 coverage gap belongs in gate 2 and only there. It is a product judgment about whether a
costly estimator earns its place, and it has no theorem behind it.

### Four rules that make the gates operational

**"Compatible with 0.95"** means

```text
|coverage-hat − 0.95| ≤ 1.96·sqrt( p-hat(1 − p-hat)/M )
```

with a Wilson or exact binomial interval preferred near the boundary — and with a **minimum
replication count and the reported interval width**, so that a very wide Monte Carlo interval
cannot be read as success.

**Coverage need not deteriorate monotonically in `n`.** When `α < 1/2` and the drift coefficient
is nonzero the root-`n` bias eventually grows, but finite-sample coverage can still be
non-monotone: the variance changes with `n`, targeting behaviour changes, truncation binds at
different rates, reduced-regression error changes, and higher-order terms can temporarily offset
the drift. **Use the slope or trend of the root-`n` bias and the remainder as the primary rate
evidence**, and treat a monotone coverage decline as supportive rather than mandatory.

**The 2% invalid-fit threshold is a product choice with no theorem status.** Freeze it after the
pilot and before the final study, and report the sensitivity of every conclusion to counting an
invalid fit three ways: as a coverage failure, as an exclusion with the exclusion rate reported
beside every number, and as a separate algorithm-failure outcome. **The primary report counts an
algorithmically invalid fit as a failure of the procedure** — an intention-to-treat reading. A
coverage number computed over the surviving fits is conditional on a non-random subset selected on
a diagnostic correlated with the fit having gone wrong, and reporting it as *the* coverage is the
same class of error as reporting a per-protocol analysis as intention-to-treat. The rule has to be
written down before the numbers exist, because a demonstration whose exclusion rule was chosen
after seeing which cells it helped is not a demonstration.

**Frozen at 2%, after the pilot and before the final study, exactly where this paragraph says to
freeze it.** [The pilot](coverage-study.md#what-the-pilot-measured) measured an invalid share of
`0.000` to `0.060` over 600 fits, and the bar is **left where it was** rather than raised to the
number that was seen. That is the whole discipline of this rule: `0.060` appears in one cell of
twelve — `g-drift`, Tier 2, `n = 600` — and moving the threshold to 6% so that cell passes would be
choosing the rule after seeing which cells it helped, which is the failure the paragraph above
names. So the threshold stands and that cell is a **live gate-2 risk** to be read out rather than
accommodated. Two things qualify it and neither changes it: `0.060` is 3 fits of 50, whose Monte
Carlo error is about `0.034`, so it is barely separated from `0.02` and the final study's 250
replicates are what resolve it; and **every invalid fit in the pilot was a `score` failure and none
was an `identity` failure**, so what the rate measures is fits that did not converge rather than a
software defect — which is the distinction gate 1's clauses 2 and 3 are worded apart for, and it is
now a column rather than an inference.

**A mixed cell's coverage number is reported pooled, with the two contract populations beside
it.** This is the fourth rule, it is [piece C3](../roadmap.md#c-the-demonstration)'s decision
taken before its dispatch as clause 0 requires, and the written reason is
[C1's witness](coverage-study.md#what-tier-1-already-showed-and-it-is-not-what-the-design-expected):
cells are **mixed** rather than pure, so *"the study's cells are inside the contract"* is a share
and there is no cell-level label for a coverage number to be read under.

- **The primary number pools every fit in the cell**, which is the estimator as shipped, and it
  is the same intention-to-treat reading the rule above it already takes. Clauses 5 and 6 read
  it, and clause 0 reads the **share** beside it.
- **Coverage within the theorem-side and the bound-active draws is reported next to it, as
  description and not as a verdict.** The contract label is a *post-fit property of the draw*,
  so conditioning on it selects a non-random subset exactly as excluding invalid fits does —
  and the objection is the same one, so it earns the same answer. **Neither stratum may be
  quoted as "the theorem-backed estimator's coverage"**, in this study or in any document
  reading it; doing so is [stop-ship 15](../roadmap.md#stop-ship) with a number attached.
- **What the strata are for** is the one question the share cannot answer: whether the two
  populations behave differently at all. A cell whose two strata agree says the constrained
  rendering is not doing anything visible to coverage at this size; a cell whose strata diverge
  is a finding, and one neither the pooled number nor the share would show.
- **The label stays out of every verdict**, which is why `CorrectionCheck.passed` does not read
  it and why no accounting here does either. A bound-active fit can have every identity at
  `1e-17` and every score negligible — on the well-overlapped draws C1 measured, it does — so
  folding the label into a pass/fail would report a sixth of sound draws as broken.

`benchmarks/drtmle_coverage.py::stratum_rows` is the implementation and the harness prints it
under that caveat; the rule is here and not there, for the reason the design note gives about
every rule in this study.

## 6. What each new test has to be watched to fail

[Lesson 4](investigation-log.md#what-the-sizings-got-wrong) is that a test written after a
change and never watched to fail pins nothing, and
[lesson 2](investigation-log.md#what-the-sizings-got-wrong) is that this variant's
instruments go blind in a place that can be named in advance. So the mutation goes in the plan
rather than being found afterwards.

| layer | what it pins | the mutation it must fail against |
| --- | --- | --- |
| unit | `Q_r`, `g_{r,1}`, `g_{r,2}` are the three definitions | swap `gr1` and `gr2` |
| unit | the corrected curve is a **difference** | add the corrections instead of subtracting |
| unit | the mechanism correction's **sign** matches the adjudicated theorem (item 21) | flip `D_A`'s sign and watch the variance move materially — on a fixture with nonzero `Q_r` |
| unit | the curve reads *starred* nuisances | read the initial `g` or the initial reductions |
| unit | arm indexing | swap the arm columns |
| unit | each of equations (8), (9), (10) is solved | drop one equation at a time |
| unit | the stored eq (9) score **equals** `mean(w·D*_g)` at the returned state (item 20) | **run**: recompute the "stored" score instead of reading the record, and the identity holds everywhere |
| unit | the stored eq (10) score equals `mean(w·D*_Q)` at the returned state | swap the `gr2/gr1` ratio |
| unit | the identity holds on a fit where the bound **binds** | **structural**: the fixture's two draws clip 0 and 5 rows and the test asserts both, so moving it breaks the test rather than silently passing |
| unit | `B_clip` reproduces the mismatch on the current implementation | **run**: zero the diagnostic, and the residual goes unexplained |
| unit | the identity is checked **per arm** before the contrast | **run**: swap the arm columns when reading the stored score; and separately, a hand-built pair of per-arm biases that cancel exactly in the contrast |
| unit | the identity carries the row **weights** | **run**: drop `w` from the means, and swapping a fitted result's weights stops moving them |
| unit | the rows are on **one** outcome scale | **run**: drop the `scaler.range` factor, and the estimand's curve no longer equals its arms' corrections |
| unit | the split into `D*_g` and `D*_Q` did not move the curve | **run**: return their difference from `total()`. Watched first against a test comparing `total()` with `reduced_corrections` — which **passed**, because the second calls the first, so that test was replaced with one comparing each half against longhand |
| unit | an identity failure reaches the report (item 16's machinery) | **run**: mark the identity rows passed, and both the verdict and the summary go quiet |
| unit | the stopping rule accepts either ruler | delete the absolute branch (already done, item 12) |
| unit | an oracle reduction tracks the **targeted** pair, not the initial one | **run**: have the injected `refit` close over the initial nuisances and ignore its argument — six tests red, including the truth recovery, because the reductions stop being the ones equations (9) and (10) are stated at |
| unit | the oracle reduction *is* what the saturated learner estimates | **structural**: the two are compared array for array at the exit of two independently run alternations, so an oracle computing something else smooth cannot agree by luck |
| unit | a fluctuation's recorded `score` **is** `score_columns` at the state it returned | **run**: score on the *weighted fit* submodel rather than the scoring one — reddens only the `target_weights=True` cases, which is why they are parametrised. The obvious mutation, recording the loop's in-loop score, was run and is **inert**: it is taken after the step, at the iterate the loop returns |
| unit | every round reads equation (8) at the state it **exits** at, not the state it was solved at | **run**: delete the restatement — before it was made unconditional this reddened nothing but its own call-site pin, 68 of 69 tests passing, which is [lesson 12](investigation-log.md#what-the-sizings-got-wrong) |
| unit | a weighted fit transports (item 17) | reductions taken at the sampling law (already done, item 17) |
| unit | fold `k`'s nested reduced regression **trains** on fold-free arrays | **run**: hand the training rows the production design and target, and 12 of `tests/unit/test_nested_reductions.py`'s tests go red, including both agreement classes |
| unit | …and **predicts** at the production design | **run**: predict at the inner design instead. One test red — the longhand — and nothing else, which is why that longhand is written out rather than trusted |
| unit | it reads **its own** fold's copy | **run**: read fold `k+1`'s. Caught by the call-site pin and by nothing else, because every other fixture here uses copies that are equal across folds; the fixture's per-fold `spread` exists for this mutation alone |
| unit | the fold-free arrays are moved by the fluctuation the production ones took | **run**: drop the `carry` in either funnel, and the degenerate control diverges after the first refit. That control — inner copies set *equal* to the production arrays — is what pins the whole transfer in one equality: `psi`, the variance, the curve, every `epsilon` and the round count, bit for bit |
| unit | a saturated reduction on a finite law **cannot** see the construction in `g_{r,1}` | **run, and it must *pass***: both designs induce the partition by `W`, and `g_{r,1}` is the one reduction whose target is data rather than an estimate. `TestASaturatedReductionOnAFiniteLawCannotSeeItInGr1` is that kept as a test, so a later reader finds a named degeneracy rather than a defect |
| unit | the `reseed` yardstick answers for the arm it was asked about | **run**: hard-code `"paper"` inside `route_rows` again, and the `nested` count reads `0/2` where it should read `2/2` |
| oracle | the drift decomposition | delete one correction term |
| component | each object of the curve equals what the derivation gives for it, **at a value where it does not vanish** | perturb one component and watch only that row move; and evaluate the whole checklist at the *truth* instead, where every row passes and the check is vacuous — the mutation that says the law is misspecified on purpose |
| unit | the corrected curve **is** the Gateaux derivative in each half of the union model (A1a) | **run**: negate `CorrectionParts.total()` under each guard separately — `g_right` goes red on one and `q_right` on the other, so the two cells are not one test twice; and swap `gr1`/`gr2` inside `fit_reduced`, which reddens `q_right` only |
| unit | that comparison is not vacuous | **run, and it must *pass***: the same module at `law.G`, `law.Q`, where both corrections are zero row by row and every assertion holds under *either* sign. `TestTheControlsBite::test_at_the_truth_the_whole_comparison_is_vacuous` is that mutation kept as a test |
| unit | the targeting step is what centres the curve | **run**: skip the equation-(8) fluctuation, and the corrected curve is the efficient one plus `0.11` at arm 0 and `0.125` at arm 1 — pinned from both sides, by asserting the targeted plug-in *is* the truth and the untargeted one is not |
| unit | the reported interval is Theorem 1's `σ²_n`, not merely the curve (A1a) | **run**: negate one term inside `total()` and watch `TestTheReportedVarianceIsTheorem1s` go red while the array-level `test_the_parts_are_the_theorems_positive_terms` stays green — the separation that says the two tiers are different claims; and diagonalise `influence_covariance`, which reddens the contrast row alone |
| integration | `guard=()` is `TMLE` bit for bit | route the empty guard through the reduction loop |
| integration | each guard removes its own direction | cross the guard semantics |
| unit | only the guarded equation's correction is **in the curve** (item 23) | **run**: drop the branch from `CorrectionParts.total()`, and cross it — `tests/unit/test_influence_drtmle.py::TestOnlyTheGuardedEquationsCorrectionIsInTheCurve`, whose fixture is checked at the exact law too, where all three guards agree and every array is zero |
| unit | the guard *travels* to the corrections rather than being read twice | **run**: have `correction_parts` pass a literal `("Q", "g")` — fails the production-path tier and not the array tier, which is the separation |
| integration | a partial-guard fit's unguarded correction is reported and **not judged** | **run**: hold it to the bar and watch a correct fit fail; and revert `total()` to the sum and watch the same fit's curve decentre — `TestASingleGuardSubtractsOnlyTheCorrectionItSolvedFor` |
| integration | a failing score check is visible in `summary()` (item 16) | silence the verdict (already done, item 16) |
| unit | item 25's label can read **either** value (C1) | **run**: hard-code `contract` to `"theorem"`, and `TestTheContractSaysWhichEstimator`'s pinched fit goes red. The label reading one value on every fit is [stop-ship 14](../roadmap.md#stop-ship) a third time, so the class fits one of each and asserts they disagree |
| unit | the label is a **scope** column and not a verdict | **run**: have `CorrectionCheck.passed` read `truncations_active`, and a bound-active fit whose identities all hold is reported as broken |
| unit | `initial_clipped` reads the **untruncated** initial mechanism | **run**: clip it first, and the count is zero on every fit — the column that could not disagree, in the place it is easiest to reintroduce |
| unit | `gr1_margin` reads the **untruncated** `gr1`, so it is signed | **run**: read `bounded_gr1` instead, and the pinched fit's negative margin becomes zero |
| unit | the estimator receives the prescribed sequence, on the declared scale (C1) | **run**: drop `q_bounds=` from the shared settings and the scaler is the draw's own range — which is `OracleOutcomeContinuous`'s `O(n^(-1/2))` recovery error, *the same order as the injected drift* |
| unit | the reductions stay **fitted** rather than prescribed | **run**: omit the reduced learners and the fit raises — on the *width* of a univariate design rather than on a check, which is why the harness names them explicitly instead of relying on it |
| unit | the shortfall is paired **on the draw** | **run**: pair by position, and a missing arm makes the difference draw-to-draw variation wearing the estimator's name |
| unit | the primary accounting counts an invalid fit as a **miss** | **run**: count coverage over all rows regardless of `valid`, and an invalid fit whose interval happens to contain the truth is scored as a success |
| simulation | the drift coefficient is nonzero as designed | set `h_a` orthogonal to the misspecification weight and watch `TMLE` cover anyway. **The construction makes this structural**: `h_a` is aligned with the weight and normalised by the quadrature defining `c_a`, with the arms given opposite signs, so `tests/unit/test_drtmle_coverage.py` asserts the realised coefficients *are* the declared ones and that `c_1 > 0 > c_0` |
| simulation | slow `Q̄`, wrong `g` | `TMLE` must under-cover, or the regime was not entered |
| simulation | slow `g`, wrong `Q̄` | as above, in the mirror cell |
| simulation | both nuisances right | no material efficiency loss from the corrections |
| simulation | both wrong | no false robustness claim |
| simulation | `√n·R_remaining → 0` (item 13) | freeze the reductions at their initial fit |
| simulation | `R_Q` and `R_g` are reported separately | report only the total, and construct a DGP where they cancel |
| stress | weak overlap | inference must be marked invalid where the scores fail |
| stress | repeated splits (item 18) | reuse one draw's reductions in every draw; drop a draw from the average (already done, item 18) |

**Two of these cannot be written against the exact law**, and the reason is derivable rather than
empirical: everything the variant adds vanishes there row by row. The corrected-curve rows want
nuisances that are wrong on purpose, which is what `tests/unit/test_remainder_drtmle.py`,
`tests/unit/test_influence_drtmle.py` and `tests/unit/test_influence_gateaux_drtmle.py` all do.

**And a mutation that must be watched to *pass* is worth as much as one watched to fail**, which
is the row above about running the Gateaux module at the truth. A test suite records the mutations
it caught; what it almost never records is the ones it cannot, and those are what a later reader
mistakes for coverage. Four of them are named in that module's docstring — item 23, equation
(9)'s covariate sign, a reduced regression's pooling weight, and the cross-fitting construction —
the first two measured by running the mutation rather than reasoned about, and each with the
module that *does* cover it, or with the piece that still owes it. Add to that list when a new instrument lands
here; do not quietly narrow a parametrisation instead.

**The fourth of those now has its module, and the reason it was blind was not the one recorded.**
`tests/unit/test_nested_reductions.py` is what covers the cross-fitting construction, and building
it found that the stated reason for `test_influence_gateaux_drtmle`'s silence — "at a saturated
reduction every conditioning cell is a singleton" — is wrong twice over. On that law the design
takes three values over a thousand rows, so the cells are not singletons; and saturation of the
*reduction* is not what decides it, since under a primary learner that learns any reduction learner
returns different arrays. What makes that module silent is `cross_fit=False` and oracle primary
learners: one fold has no complement to nest inside, and a learner that ignores its training rows
returns the same function whichever rows it saw. The conclusion is unchanged and the reason is now
the true one — which matters, because the false one would have licensed reading a *cross-fitted*
saturated fit as evidence about fold reuse. [Stop-ship 14](../roadmap.md#stop-ship) carries the
corrected wording, and the corrected statement is now asserted rather than described.

## 7. The cross-fitting construction (piece A1b)

Item 15 asks whether the reduced regressions' **pooled** cross-fitting satisfies the
empirical-process conditions of the DRTMLE expansion. [The concordance's
§8](theorem-concordance.md#8-cross-fitting-is-not-in-the-sources-and-the-argument-for-it-item-15) carries the argument; this
section carries what a run can say about it, and the rule that run is read under.

**What the argument needs measured, and it is not the thing a comparison usually measures.** The
argument splits fold `k`'s empirical process into a term that is conditionally mean zero — the
*nested* construction's — and a residual `(P_n − P_0)Δ_k`, where

```text
Δ_k = (fold k's reduced regression, as fitted) − (the same regression, fitted fold-free)
```

Given a bounded-entropy univariate class the residual is `o_p(n^(−1/2))` **provided `‖Δ_k‖ → 0`**.
So `Δ_k` is not a diagnostic of the comparison, it *is* the assumption; and
`reduced_crossfit="nested"` is the only thing that computes it.

**The direction is the opposite of §4's and that is the whole reason to freeze this in advance.**
There, two update orders provably solve the same three equations, so agreement was the expected
finding and a persistent difference was the surprise. Here the two constructions are genuinely
different estimators, and what the pooled argument needs is not that they agree but that the gap
**shrinks**. A large but shrinking difference is the "pooled is fine" outcome; a small but stable
one is not. Choosing that reading after seeing the numbers would be exactly the failure the frozen
rules exist against.

### The cross-fitting rule, frozen before the dispatch

**It may be changed before the dispatch with a written reason, and not after it.**

> **The pooled construction is the one [C](../roadmap.md#c-the-demonstration)'s final dispatch fits
> if:**
>
> 1. the median `|Δψ|/se` between `nested` and `base` **decreases across the sizes** on both
>    processes;
> 2. the median `se` ratio is inside `[0.95, 1.05]` at the largest size on both processes;
> 3. the **count** of draws in which the construction difference exceeds the *reseed* difference is
>    at or below half the pairs at the largest size — the construction moving `psi` no further than
>    a redrawn split of one construction does;
> 4. no fit in either arm fails its score check or its state identity.
>
> The **primary** evidence is clause 1's slope. Clause 3 is supporting and is stated that way
> *before* the run rather than after it, because [B2b](investigation-log.md#the-same-rule-at-thirty-six-draws-and-why-the-two-readings-are-not-nested)
> measured the identical count clause underpowered at twelve draws and again at thirty-six; writing
> it as primary here would be repeating a mistake this page has already recorded. Clause 3 is also
> **one-sided** from the outset — a count far below half is evidence *for* the conclusion — which is
> one of the two restatements §4 says may be made before a further dispatch and never after one.

**What would falsify it**: a construction difference that **does not shrink** while the reseed
difference does. That says the two constructions converge to different limits, that `Δ_k` is not
`o_p(1)`, and that Theorem 1's expansion is not available for the pooled construction at the
reduction learner in use. The nested construction then becomes the reference,
`reduced_crossfit=` changes default, and C's dispatch is rerun — which is the rework edge
[the roadmap](../roadmap.md#a1b--the-cross-fitting-construction) already prices.

**Three readings this rule deliberately refuses.**

- **A `weak-overlap` difference is not evidence about item 15 on its own.** Two of the three
  reductions condition on `ĝ`, `g_{r,2}`'s target is a quotient by it, and a third of that
  process's `(row, arm)` pairs clip at the initial mechanism. So the two arms there differ for a
  truncation reason as well as a construction reason. Read `clip share` and `min gr1` beside it.
- **`library="rich"` is outside the entropy condition by declaration**, so a `rich` cell would
  measure a construction whose pooled argument was never claimed. The sweep runs at `glm`, which is
  inside — see the concordance's §15 rows.
- **A round-count or wall-clock difference is not a validity finding.** The nested arm runs 1.3x to
  17x the base arm's wall clock and reaches the outer cap more often, because its reductions are
  noisier and equation (10)'s solve is near-singular by construction
  ([limitation 4](../roadmap.md#limitations-recorded-rather-than-fixed)). Clause 4 is what that
  bears on, and clause 4 is about the *scores*, not about how many rounds it took to solve them.

**The dispatch ran and the rule did not resolve; the numbers and the reading are in [the
log](investigation-log.md#what-the-a1b-dispatch-measured).** Clauses 2, 3 and 4 pass. Clause 1
passes on `nonlinear` and fails on `linear` — where the failure takes the falsifier's literal shape,
the nested difference staying flat while the control halves, and where the nested difference is
nonetheless 3 to 7 times *below* that control at every size. The rule is **left exactly as frozen**:
it was written to be evaluated, it has been, and it came back unresolved. Rewriting it now to fit
what the numbers did is the one thing this section forbids.

**What the run establishes about the *rule* rather than the estimator.** A median over twelve draws
will not carry a slope claim — the second time this page has measured that, after the update order
at twelve draws and again at thirty-six — and the pre-registered restatement (read the *ratio* of
the two medians) does not rescue it, since neither process's ratios are monotone either. The next
dispatch should not buy more seeds at these sizes. It should build the instrument the next paragraph
records, because clauses 1–3 all read a *consequence* on `ψ`, and a consequence is where
cancellation and split noise live.

**What is not on this list, and could be.** A direct estimate of `‖Δ_k‖` — the paired `L₂` distance
between the two arms' reduced arrays, which are row-aligned by construction since the arms share a
draw and a fold seed. Clauses 1–3 measure its *consequence* on `ψ` and `se`, and a consequence can
hold by cancellation. Reporting the norm itself would need each fit's `(n, 3K)` reduced block kept
and paired after the sweep; it is a `--keep-reductions` flag and a table, it is not research, and it
is the sharpest instrument this section could have. It was recorded rather than built because the rule
above is evaluable without it — and having evaluated it, **this is now the piece of work the next
dispatch is**, rather than an optional sharpening.
