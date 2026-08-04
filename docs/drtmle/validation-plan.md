# DRTMLE: the validation plan

What will be run, on what fixtures, against what frozen decision rules. [The
roadmap](../roadmap.md) says which pull request lands which of these and in what order; this file is
the detail those pull requests are executed from, so that a rule is written down before the number
it judges exists.

Four sections match four pieces of work: [B1a](#1-the-invariants-piece-b1a) is the identity patch,
[B1b](#2-the-targeting-candidates-piece-b1b) is the targeting decision,
[A1a](#3-the-component-checklist-piece-a1a) is the component checklist,
[B2](#4-the-sweep-piece-b2) is the convergence and overlap sweep, and
[C](#5-the-controlled-study-piece-c) is the demonstration. [The mutation
table](#6-what-each-new-test-has-to-be-watched-to-fail) is what makes any of it evidence.

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

### Evaluating `P₀D̂`, which is not automatic for a cross-fitted fit

**Never substitute `P_nD̂`** — that is the quantity targeting drove to zero, so it answers a
different question. But out-of-fold prediction arrays at the observed rows do not define functions
on new `W`, so integrating `P₀D̂` needs the fold-specific primary nuisance models, the
fold-specific reduced-regression models, the fold-specific targeted transformations, and a stated
convention for averaging or conditioning over folds.

- **Tier 1**: exact finite-support summation or a very large independent draw. No model retention
  is needed, because the nuisance sequence is prescribed.
- **Tier 2**: add a **benchmark-only** fitted nuisance object exposing `predict(new_data)` per
  fold; evaluate each fold's corrected curve on an independent draw using the nuisance functions
  trained for that fold; average the fold-conditional `P₀` values with the same fold weights the
  estimator uses. A completely independent training/evaluation split is the alternative.

**Document the conditioning convention.** Without it `R_remaining` can be an artefact of how
fold-specific fits were extrapolated to the integration sample rather than a property of the
estimator.

### Reporting `R_Q` and `R_g` separately

The single `R_remaining = ψ̂ − ψ_0 − (P_n − P_0)D̂_DR` is necessary and not sufficient: a total
trending to zero can conceal cancellation between the two appendix branches. Where the DGP
permits, report `R_Q` and `R_g` separately, their component products, their signs, and the total.
See [the concordance's §5](theorem-concordance.md#5-the-remaining-remainder-terms) for the
exact terms.

### What to report

Per estimator, per cell, per size, per seed batch: bias, `√n` bias, empirical sd, mean estimated
`se` and their ratio, coverage, interval width, rejection rate under a null variant,
targeting-failure rate, the share of intervals marked invalid, the correction terms' own means,
variances and covariance, both remainder diagnostics and the two branches, the nuisance errors and
their slopes, the realised drift coefficients, elapsed time, and a Monte Carlo standard error
against every one of them.

**And the three truncation witnesses**, which are gate 1's new clause 0 and are cheap because
`benchmarks/bench_drtmle.py` already computes two of them: `clip share` at the initial mechanism,
`margin` at the exit, and `min gr1`. They are not diagnostics of a fit going wrong — a bound-active
fit can have every identity at `1e-17` and every score negligible, and on `weak-overlap` it does.
They say which *estimator* the row is evidence about, which is a different question and one no
other column on this list answers.

### Sizes and replications

At least three sizes — `600 / 1,200 / 2,400` is the shape, adjusted upward if the prescribed rate
is not visible — and a pilot of 50 to 100 replicates per cell before anything is frozen. The
frozen study wants **250 at minimum and 500 if the budget reaches**: at a true 0.95 the Monte
Carlo standard error of a coverage estimate is `0.014` at 250 and `0.010` at 500, so 250 resolves
a `0.95`-against-`0.88` gap comfortably and does not resolve `0.95` against `0.93`. Then an
independent second seed batch, run after the first is complete. Changing sizes or counts *after*
seeing coverage is permitted only as a new experiment, documented as one.

**What it costs.** A `DRTMLE` fit is 43s at `n = 1,200` (measured,
[the sweep](investigation-log.md#how-the-alternation-exits)) and a study runs both
estimators over every replicate. Two cells by two sizes by 250 replicates is ~2,000 fits, which is
~24 hours serial and about two on a 12-way `matrix:`. A third size and the nuisance-rate columns
roughly double it. That is a dispatch-only workflow of its own — `drtmle-convergence.yml` is the
template — and the nightly tier must not absorb it.

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

### Three rules that make the gates operational

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
| simulation | the drift coefficient is nonzero as designed | set `h_a` orthogonal to the misspecification weight and watch `TMLE` cover anyway |
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
