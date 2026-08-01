# Roadmap

Two lists, and they are different kinds of thing. **Variants** are estimators that plug into
the shared base classes (`estimators/base.py`, `inference/`, `learners/`, `fluctuation/`).
[Refusals worth lifting](#refusals-worth-lifting) are parameters this package already has the
machinery for and has simply not written down — drawn from [Not written
yet](methodology.md#not-written-yet), which is the full list of candidates rather than the
chosen ones. **All six have landed**, and so has the second variant, so both lists are now a
record of what was done and what the sizing got wrong rather than a plan. What is left is
named at the end of [What `drtmle` touched](#what-drtmle-touched): one prerequisite that has
to happen outside this repository, and one claim nothing here has yet demonstrated.

## Variants

- **longitudinal TMLE (`cleverly.longitudinal.LTMLE`) — landed**, for static regimens and
  for **dynamic rules** `d_t(H_t)`, with time-varying confounding and monotone censoring,
  for a **survival outcome** — one absorbing event indicator per node, reporting the
  cumulative risk curve with joint bands over it — and for **competing risks**, where more
  than one absorbing state per node makes the report a cause-specific cumulative incidence
  per cause; see [Treatment given over time](user-guide.md#treatment-given-over-time),
  [A survival outcome](user-guide.md#a-survival-outcome) and [Competing
  risks](user-guide.md#competing-risks). What
  it still refuses is listed there under a `kind` column. The largest thing it will go on
  refusing is the *other* competing-risks estimand — the incidence under **elimination** of
  the competing events, which intervenes on them rather than conditioning on the history,
  and so is [a different question](methodology.md#a-different-question) rather than a gap: a
  further factor per node in the denominator, and its own identification. A working model over
  regimens was the largest thing it was *missing* as against refusing; that is item 4 below
  and has landed
- **doubly-robust nonparametric inference (`drtmle`)** — van der Laan (2014); Benkeser,
  Carone, van der Laan & Gilbert (2017); Benkeser & Hejazi (2023). Every interval reported
  here is valid when the second-order remainder is
  negligible, which needs *both* nuisances converging fast enough; `drtmle` buys an interval
  that stays valid when only one of them is consistent, by estimating additional
  reduced-dimension regressions of each nuisance's residual on the other and solving their
  score equations too. That is a genuine variant rather than a further estimand, so it plugs
  in at `TMLE._nuisances` and the targeting step rather than at the target registry — which
  is right as far as it goes, and is two of the **six** seams it turns out to touch. It
  predates the six below and was sized from the paper rather than from a read of what would
  have to change here; [What `drtmle` touched](#what-drtmle-touched) is that read,
  and now also a read of the source — the derivation that section left open is
  [pinned](#what-the-source-settles) rather than conjectured. **Landed**, as
  `cleverly.DRTMLE`: the remainder module, the reduced-dimension regressions, the
  three-equation alternation, the estimator and the influence curve, in four commits. Two
  things it does *not* come with, and both are stated where a reader meets them rather than
  here alone: the curve's form is read off `drtmle`'s implementation and **Theorem 1 of
  Benkeser et al. (2017) is still unread**, and a coverage study on the off-diagonal of the
  misspecification grid found **no gap for the variant to close** at the sizes it could
  reach. So what has landed is the estimator; what has not is evidence that it buys what it
  is for

## What `drtmle` touched

The read the bullet above was missing, taken against `estimators/`, `fluctuation/`,
`inference/` and `tests/` rather than against the paper. It was written **before** any of the
work, which is the point: each of items 1 to 6 below records what its sizing got wrong, and
the misses here are ones that could be named in advance rather than found by mutation
afterwards. It is kept in the tense it was written in, with what each seam actually cost
marked where it differed — a section rewritten after the fact to match the outcome would be
worth nothing to the next sizing.

It has since been read *against the source* as well, and the two halves are kept apart
deliberately. [What the source settles](#what-the-source-settles) is the derivation, which
this section used to leave open under a heading called "three things to pin"; two of those
three are now answered and the third is still open, and saying which is which is the point of
keeping the heading's shape. What the source changed about the **plan** — the scope, and two
further seams — is folded into the sections below, each marked where it moved.

### What the source settles

Read against Benkeser & Hejazi (2023) — `docs/pdf.pdf` in this repository, the software paper
for the R package — and, where that paper defers, against the package itself. The two are worth
keeping distinct: the paper states the estimating equations and explicitly leaves the influence
function to the 2017 Biometrika paper, so one of the three answers below comes from the
implementation rather than from the literature, and is labelled where it does.

Written in this package's notation, with `1_a = 1{A = a}` (times the complete-case indicator
under `delta=`). Three reduced-dimension regressions, each *defined* relative to a given
outcome regression `Qbar-hat` and mechanism `g-hat` — which the source's algorithm then
updates along with them, a point that becomes [seam 5](#six-seams-where-the-sizing-names-two)
and is easy to miss on a first reading:

- `Qr(a, w) = E[ Y − Qbar-hat(a, W) | A = a, g-hat(a|W) = g-hat(a|w) ]`, the *reduced outcome
  regression*: a univariate regression of the outcome residual on the estimated mechanism,
  fitted on the rows with `A = a`.
- `gr1(a | w) = P( A = a | Qbar-hat(a, W) = Qbar-hat(a, w) )` and
  `gr2(a | w) = E[ {1_a − g-hat(a|W)} / g-hat(a|W) | Qbar-hat(a, W) = Qbar-hat(a, w) ]`, the
  two *reduced mechanisms*, both univariate.
- Or, in van der Laan (2014)'s original form, one **bivariate**
  `gr(a | w) = P( A = a | Qbar-hat(a, W), g-hat(a|W) )` in place of that pair. Benkeser et al.
  split it in two on the argument that two univariate regressions are easier to estimate
  consistently than one bivariate one, and `reduction="univariate"` is `drtmle`'s default.

Three score equations, and **two nuisances fluctuated**:

| | equation | fluctuates |
| --- | --- | --- |
| (8) | `Pn[ 1_a / g*(a\|W) · (Y − Qbar*(a, W)) ] = 0` | `Qbar`, by today's `mean_submodel` |
| (9) | `Pn[ Qr(a, W) / g*(a\|W) · (1_a − g*(a\|W)) ] = 0` | **`g`** |
| (10) | `Pn[ 1_a · gr2(a\|W) / gr1(a\|W) · (Y − Qbar*(a, W)) ] = 0` | `Qbar`, a second covariate |
| (10′) | `Pn[ 1_a / gr(a\|W) · {gr(a\|W) − g*(a\|W)} / g*(a\|W) · (Y − Qbar*(a, W)) ] = 0` | `Qbar`, under `reduction="bivariate"` |

**The first of the three open questions is closed, and so is the doubt about termination.**
One of the extra equations does fluctuate `g`, so the targeting is an alternation and
`solve_with_mechanism` is what it resembles — and the argument that makes *that* loop
terminate carries over, which is the part worth stating rather than leaving to be discovered.
(9) is a weighted logistic MLE of `A` given `W`; (8) and (10) are the outcome
quasi-likelihood. Those are separate factors of the likelihood of `(A, Y) | W`, so each step
maximises its own factor with the other held fixed and the joint value never decreases —
exactly the reasoning `fluctuation/mechanism.py` already writes out, and exactly the
reasoning that did *not* carry over to `solve_with_projection`, whose second half is a
least-squares solve rather than a likelihood. One difference from the mechanism alternation
is worth carrying into the implementation: (9)'s covariate is `Qr/g*`, which reads the very
`g*` it fluctuates, where `ipsi`'s mechanism covariate reads only the targeted `Qbar*`. Each
round is still an MLE in a submodel *through the current point*, so the monotonicity survives
as a statement about a path rather than about one submodel — and `drtmle` caps its outer loop
at three rounds, which is not what an implementation does when it is relying on convergence.

**The second is closed by the implementation rather than by the paper**, which says the
influence function is "available in Theorem 1 of D. Benkeser et al. (2017)" and gives no
formula. `drtmle` reports

```text
D = D* − D*_Q − D*_g,
    D*_g = Qr(a, W) / g*(a|W) · (1_a − g*(a|W)),
    D*_Q = 1_a · gr2(a|W) / gr1(a|W) · (Y − Qbar*(a, W)),
```

and takes the covariance of that. **Minus**, where this section offered "`D*` plus the extra
components" as the alternative. All three empirical means are driven to zero by the
targeting, so the subtraction cannot move the point estimate; it moves only the variance,
which is the whole of what the variant buys. One trap for anyone reading the R source
alongside the paper: `grn1` there is the paper's `gr2` and `grn2` is the paper's `gr1` — the
numerator and denominator roles are swapped between the two, so a formula transcribed from
one and checked against the other is inverted and still plausible.

This one is a **fidelity claim about `drtmle`, not a theoretical result**, and the distinction
matters for how it should be checked. Theorem 1 of Benkeser et al. (2017) is where the
influence function is derived, and it has not been read here; what is written above is what
the package computes. If the two disagree, the theorem wins and this section is wrong — so
read it before the curve is implemented, and treat the form above as the thing to reconcile
against rather than as the specification.

**The third is genuinely open and is the only one.** Nothing in the source addresses how the
reduced regressions are cross-fitted, and the difficulty this section named — that their
*design* is itself an out-of-fold prediction — is real and unaddressed there. It stays
[below](#the-one-thing-still-to-pin--settled-on-an-argument-rather-than-a-measurement) as the one decision to make before any code.

The source also volunteers this section's own instrument finding, in its Discussion, as
advice about the reduced learner library: "when the OR and PS are consistently estimated, the
reduced-dimension regressions are identically equal to zero". That claim was derived here
before the paper was read, and the authors state it outright.

### Six seams, where the sizing names two

Four were nameable from this codebase alone. Two more are visible only once the source pins
the algorithm, and each of those is a place where a decision has to be *stated* rather than
inherited.

1. **`TMLE._nuisances`, as `CTMLE` does — landed, as far as the arrays go.** *Adding* fields to
   `NuisanceEstimates` rather than replacing `propensity`, which is the difference between the
   two variants and the reason this one cannot be "override `_nuisances` and let the inherited
   `retarget` do the rest". The reduced-dimension regressions do go through
   `cross_fit_predictions` untouched — `estimators/_nuisance.py`, not `learners/crossfit.py` —
   with a one-column design, a residual target and `fit_mask` for the arm's rows. There are
   **three** of them per arm rather than the two this said, and under `reduction="bivariate"`
   two, one of which has a two-column design; the bivariate reduction is refused by name for
   now.
   `estimators/reduced.py` holds a `ReducedSet` and `fit_reduced`; `NuisanceEstimates.reduced`
   is the field; `tests/unit/test_reduced_regressions.py` checks the fitted values against
   `test_remainder_drtmle`'s longhand arithmetic rather than against a second derivation.
   **No estimator reaches any of it**, deliberately: a `DRTMLE` that fitted the reductions and
   reported the ordinary estimates would be this variant's whole failure mode, so the name
   arrives with the alternation — and in the event with the *curve*, one commit later still,
   because an estimator that solved the equations and reported the plain interval would be
   that same failure mode for the length of a commit. Three things the sizing did not name.
   *Where it is built is a departure from the `shifts`/`incremental` precedent and had to be
   argued rather than copied.* Those are built **inside** `fit_nuisances`, so that "the tilt
   and the `g` it tilts came from one out-of-fold model" is structural. This is built outside,
   in a `_nuisances` override, because it belongs to one variant rather than to every fit —
   and the invariant survives only because `fit_reduced` takes a whole `NuisanceEstimates` and
   reads `folds` off it, so it cannot be handed a mechanism and a split that did not come from
   one construction. That is why it takes the object rather than two arrays.
   *One bound is chosen at fit time, and it is the only one in this package that is.* `gr2`'s
   **target** is a quotient by the mechanism, so it cannot be stored raw and re-truncated by
   `retarget` the way `bounded_propensity` and `bounded_missingness` are: the array *is* a
   regression of that quotient. It divides by the fit's own declared `g_bounds`, which travels
   on `ReducedSet.g_bounds` — and the consequence has to be said where a reader meets it:
   `truncation_curve` moves the clever covariate's denominator and **does not** move these
   arrays, so the part of the curve that comes from the extra equations is flat by
   construction. That is item 5's lesson arriving from the other direction, and flat by
   construction reads as insensitivity rather than as a limitation. The *design*, by contrast,
   is the untruncated `ĝ`: truncation is for denominators, and bounding a conditioning
   variable would collapse the extreme rows into ties and coarsen the very σ-algebra the
   reduction projects onto.
   *And the exact law has a second blind spot, in the same place item 4 found its first.* Its
   outcome is binary, so `OutcomeScaler` is the identity and `scale` is a no-op — taking `Qr`'s
   residual against the **raw** outcome rather than the scaled one was applied and passed all
   25 tests of the new module. An affine relabelling of the outcome is what catches it, exactly
   as `tests/e2e/test_ltmle_msm.py` catches the same mistake for the working model over
   regimens. Of seven deliberate mutations that one was the only survivor, and it did not
   announce itself the second time either.
2. **The targeting dispatch — landed, and the predicate is not the one this named.**
   `needs_mechanism(group)` and `needs_projection(nuisance, group)` gained a third branch,
   `needs_reduction(nuisance, group)`, and a `solve_with_reduction` beside the two solvers in
   `estimators/targeting.py`. Two things came out differently. It returns **two** values, not
   the re-derived `NuisanceEstimates` this predicted: `ipsi` re-derives because its estimand
   is a functional of `g`, and this estimand is the plug-in mean of the targeted regression,
   which reads no mechanism at all. And the predicate has to read the *nuisances*: the group
   is still `"mean"`, so one keyed on the group name would divert every ordinary fit in the
   package — which is also why neither covariate could go in `SUBMODEL_BUILDERS` or
   `MECHANISM_BUILDERS`.
   Three things the sizing did not name, and the first is the one worth carrying forward.
   *The stall rule had to watch the objective as well as the score.* The mechanism
   alternation's score falls by a roughly constant factor every round; three coupled
   equations make it **non-monotone** for the first few — measured at 2.8e-2, 2.9e-2, 1.7e-2,
   1.8e-2 before descending cleanly to 7e-9, while the joint likelihood rose at every one of
   those rounds. The score-only rule inherited from `solve_with_mechanism` stopped that fit
   at round 2 with two equations open and reported the interval anyway.
   *Every score in sight goes stale, and a stale one is invisible at the fixed point*, where
   a remembered zero and a fresh zero are both zero. So the test compares the *reported*
   score against one recomputed at the exiting pair rather than testing convergence.
   *And the scores have to be taken at the reductions the curve reads.* They were briefly
   taken at the set equation (10) was solved along while the curve read the later refit,
   which made `score_check`'s per-estimand row disagree with its per-equation rows by two
   orders of magnitude — a real defect, found by reading the diagnostic rather than by a
   test.
   *Which fixed the reporting and left the deeper half.* Making the scores and the curve
   agree does not make either of them **solved**: equation (10) is solved at the round's
   first refit and equation (9) at the previous round's second one, so with `drtmle`'s
   ordering neither is solved at the arrays the curve is built from, and the curve's mean is
   zero only insofar as the loop converged — which is the property the whole estimator rests
   on. Measured on an 800-row fit by stopping the rounds early:

   | rounds | max\|mean of reported curve\| | after the closing pass |
   | --- | --- | --- |
   | 1 | 3.7e-3 (3.5% of `se`) | 5.8e-7 |
   | 3 | 1.4e-3 | 5.7e-11 |
   | 10 | 2.6e-5 | 1.2e-10 |
   | 26 (converged) | 7.0e-10 | 3.4e-10 |

   The fix keeps the source's ordering and adds a **closing pass** that re-solves all three
   equations at the reductions the record carries, refitting nothing. Freezing the
   reductions makes the system *triangular* — `D*_g` contains no `Q̄` at all, so equation
   (9) settles first and nothing downstream disturbs it — and equations (8) and (10) are
   then solved **jointly** in one Newton step over all four columns rather than backfitted,
   which on the exact law is the difference between `3.9e-4` after twenty alternating steps
   and `1.5e-12` after one. `drtmle` has the same gap and absorbs it into `tolIC = 1/n`.
3. **`inference/influence.py` — landed**, where the reported curve gains terms the plain one
   has no analogue of. `counterfactual_means` takes an optional `corrections` mapping and is
   otherwise untouched *character for character*, because the comment already there records
   that re-associating that sum moves the last bit of every influence curve.
   One consequence the sizing did not name: `ICParts` had to gain a **third field**. It
   decomposes the curve into a positivity half and an outcome-heterogeneity half, and the
   correction is neither — leaving it out would have made the decomposition disagree with the
   curve by exactly what the variant does, and nothing in the estimation path reads it, so
   that drift would have been silent.
4. **`estimators/serialize.py` — landed.** `FORMAT_VERSION` 8 → 9 for the extra arrays, on the
   terms versions 4 and 5 were bumped. A reloaded fit that had lost them would report a plain
   TMLE's interval under the variant's name, which is the shape of mistake that bump exists
   to prevent. It landed **with the arrays rather than with the estimator that will read
   them**, which is the opposite of the order this section lists it in and is deliberate:
   `_nuisance_from` names every field it reconstructs, so one left unwritten reloads silently
   as `None`, and the reason for the bump was in hand while the block was being written. The
   cost is stated rather than discovered — every `.npz` written by an earlier version becomes
   unreadable, by the design the reader already declares.
5. **`retarget` stops being a pure function of cached arrays** — and this is the one that
   contradicts something already written down. Every targeting step here is arithmetic on
   fitted predictions; `drtmle` **fits learners inside the alternation**, re-estimating all
   three reduced regressions on every outer round against the current `Qbar*` and `g*`. That
   collides with the contract that the truncation curve, the MNAR tilt and the
   omitted-variable bound all rest on: `retarget` re-runs only the targeting step, against
   nuisances that were fitted once.
   The obvious way out — hold the reduced regressions at their initial fit, leaving `retarget`
   pure — is **a departure from the source and not a reading of it**, and the reason is one
   character: equations (9) and (10) are stated at `Qr*` and `gr*`, *starred*, and the source
   describes its algorithm as mapping initial estimates of the outcome regression, the
   mechanism **and the reduced regressions** into estimates that satisfy them. Holding them
   fixed solves a different equation, and whether that one suffices is a question for the
   theorem this package has not read rather than a matter of taste. So this is a decision with
   a cost on both sides — refit and lose `retarget`, or hold fixed and owe an argument.
   **Settled in favour of the source: the reductions are refitted on every round.** Holding
   them fixed solves a different equation, and the argument that the different one suffices
   would have to come from a theorem nobody here has read — which is not a trade this section
   was entitled to make.
   **And the cost is smaller than this feared**, for a reason that was there to be checked
   and was not: `retarget` is a *method on the estimator*, and every sensitivity path that
   reaches it already requires a live one — `sensitivity/positivity.py` and
   `omitted_variable.py` both raise when `result.estimator is None`. So the truncation curve
   and the MNAR tilt keep working; what changes is that a sweep costs about a fit per point
   rather than a fraction of one, and that a plain `TMLE` handed nuisances carrying `reduced`
   refuses by name instead of re-solving against arrays it cannot refresh. The learners reach
   the solver as a `ReductionSpec.refit` callable, so `estimators/targeting.py` stays free of
   the estimator it was separated from.
   One thing this item asked for was **not** delivered and should not be pretended otherwise:
   it wanted "a comparison of two fitted estimators". Both were run — held-fixed against
   refitted on a 2000-row fit — and they differ by about 5% of a standard error, which
   settles neither question. The decision above rests on the source's wording, not on that
   measurement.
6. **The `Submodel` column contract survives — landed, and it did.** This
   section worried that the extra covariates change what a column means, which matters because
   `sensitivity/omitted_variable.py` reads `submodel.column_for`. `drtmle`'s default
   `Qsteps = 2` is a **backfitting** minimisation — fluctuate along the second covariate, then
   along the first, rather than both in one solve, "found to be more stable in simulations".
   Taken that way each solve is a one-column-per-arm `Submodel`, `arm_columns` keeps mapping an
   arm to a single column, and `column_for` keeps meaning what it means. The worry turns into a
   reason to prefer the backfitting form, and into the reason the second covariate is a second
   `Submodel` in the same group rather than one wider one.
   What the backfitting form *costs* is the thing this section had no way to see, and it is
   worth writing down for anyone who reaches for `Qsteps = 1` to avoid it. Equation (10)'s
   covariate is `gr2 / gr1`, and `gr2` vanishes exactly where the mechanism is right — so on
   a fit whose `ĝ` is nearly right that covariate is nearly zero and its own Newton solve is
   near-singular: observed at `mean|h| = 1e-3`, `|epsilon|` reaching 280 and a singular
   Hessian in a third of the rounds on one unseeded draw. Solved *jointly* with equation (8)
   the well-conditioned `1/g` columns would dominate that Hessian, which is presumably why
   `drtmle` offers the choice at all. How often it bites was measured rather than assumed:
   across six seeded fits at `n = 800` the alternation converged in 15 to 45 rounds with no
   ill-conditioned solve and a worst score of `1e-9`, so it is a minority behaviour of
   particular draws. `ReductionFluctuation.ill_conditioned` reports it either way.

Two things follow that a reader will otherwise assume the other way. **No target is
registered**, so the [oracle-law gate](methodology.md#the-oracle-law-gate) has nothing to say
here — the report is still `ey1`, `ey0` and `ate` under those names, a different estimator
behind the same parameters exactly as `CTMLE` is, so there is no registry entry and no oracle
branch to write. The second covariate needs a builder but not `register_submodel`; a
`TargetGroup` is a plain `str` and `"sequential"` is the precedent. And **that is precisely
what makes every one of the six seams the same shape of mistake**: a reader handed a plain
TMLE's number under a doubly-robust name, with nothing in the parameter's name to say so.

### The exact-law instrument cannot see what this estimator buys

This is the finding that matters most, and unlike its two predecessors it is derivable rather
than discovered. Under a law the sample realises exactly with a saturated learner — the
setting of every `tests/unit/test_influence_gateaux*.py` module — both nuisances are exact, so
the reduced-dimension regressions that carry the extra covariates have identically zero
targets: `E[Y − Q̄₀ | A = a, ĝ] = 0` and `E[1{A=a}/g₀ − 1 | Q̄₀] = 0`. Both extra fluctuation
coefficients are then zero and the estimator reproduces `TMLE` exactly. So the package's
primary evidence that a curve is right supplies only a **degeneracy check** here, and would
pass against an implementation whose extra terms are wrong in any way that vanishes at the
truth. That is items 4 and 5's lesson arriving a third time, and the first time it has been
seen coming.

The source pins two things that sharpen this, and both make it worse rather than better. The
degeneracy is **row by row**, not merely in the coefficients: `Qr` and `gr2` are identically
zero at the truth, so `D*_g` and `D*_Q` vanish at every observation and the reported curve
is `D*` to machine precision. (Not *array for array*, which this said and which is wrong by
one bit: a saturated regression of a residual that sums to zero returns something of order
`1e-17`, and subtracting that moves the last bit. A test asserting bit-for-bit equality there
would have been asserting an arithmetic accident.) Which means the one thing the second open
question settled —
that the combination is `D* − D*_Q − D*_g` and not a sum — is invisible to every Gateaux
module there is, and needs a structural pin at deliberately wrong nuisances instead. `gr1`
is the exception that proves the shape: it is a probability and does not vanish, and it sits
in a denominator whose numerator does, so an implementation that got *it* wrong would also
pass. `tests/discrete_law.py` is the law throughout — the binary scope is what keeps it to
one, and a multi-arm widening would owe the three-armed law a branch on the rule the
oracle-law gate states for its own reasons: two arms cannot distinguish code that keys by arm
from code that has two columns and calls them 0 and 1.

What *can* see it is the remainder idiom. `tests/unit/test_remainder.py` evaluates the von
Mises expansion at nuisances that are **wrong on purpose** on the finite-support law,
deterministically and to machine precision, and this estimator's claim is precisely a
statement about that remainder — that a product of the two nuisance errors is replaced by
products of *reduced-dimension* ones. It is statable there as an equality with `TMLE`'s
product form as the negative control, which makes that module the thing to write **first**,
before the estimator rather than after it.

`tests/unit/test_remainder_drtmle.py` is that module and it has landed. Three things it
established, and the first two correct what was predicted here rather than confirming it.
**One guard removes the whole first-order remainder and two over-correct**: each extra
equation subtracts a *projection* of `R₂` — equation (9) onto `σ(ĝ)`, the other onto `σ(Q̄̂)`
— and on a three-cell law where both σ-algebras are all of `σ(W)` either projection recovers
the whole of it, so the pair leaves exactly `−R₂`. That is arithmetic rather than a defect,
since asymptotically at most one of the two errors fails to vanish and so at most one
projection is non-negligible, which is why `drtmle` solves both by default — but "the
remainder vanishes when both nuisances are wrong" is false, and a test asserting it would
have gone red and been blamed on the implementation. **The second-order claim is about the
*reduced* error**, not about the two primary ones: with exact reduced regressions the guard
removes everything at every scale, so there is no rate to measure, and what survives a
perturbed `Qr` is a product of the reduced error with a primary one — which is the whole
reason the reduced regressions may be univariate. And **a tie is what makes a reduction a
reduction**: with a distinct nuisance value in every cell the regression conditions on `W`
relabelled, so the tie constants are what separate an implementation that conditions on the
estimated nuisance from one that quietly conditions on the covariate.

**What it did not settle is either of the two open decisions**, against what this paragraph
used to claim — that whether the reduced regressions are held at their initial fit or
refitted each round (seam 5), and which folds they are fitted on, were both *measurable*
there. Neither is. The module runs no targeting step, so there is no alternation for seam 5
to be a question about; and it computes the three regressions longhand at the true law rather
than fitting them, so there is no learner and no split for a fold question to bite on. It
says so itself, in the class that measures the rate: the fitted-reduced-regression case
belongs to the stage that has learners. Both decisions moved forward one commit rather than
being answered early, and the fold one is [now settled](#the-one-thing-still-to-pin--settled-on-an-argument-rather-than-a-measurement) there.

End to end the claim is about **coverage, not bias**, and that distinction is the whole
variant. `TMLE`'s double robustness is a statement about the *point estimate*: `R₂` is the
product `‖ĝ − g₀‖·‖Q̄ − Q̄₀‖`, so one inconsistent nuisance still leaves `R₂ → 0` and `ψ̂`
consistent — which `tests/unit/test_remainder.py` already checks exactly. The interval needs
the strictly stronger `√n · R₂ → 0`. With both nuisances consistent at `n^(−1/4)` the product
delivers that; with only one, the bad factor stops shrinking and `R₂` becomes *first-order* in
the good one's error, which no nonparametric estimator drives below `n^(−1/2)`. So the
estimator stops being asymptotically linear and its coverage decays as `n` grows while its
bias does not. **`TMLE` is doubly robust for consistency and singly robust for inference**,
and closing that second gap is what this variant is for.

Which makes the nightly instrument a `CoverageStudy` over the off-diagonal of
`tests/e2e/test_double_robustness.py`'s grid, with `TMLE`'s own coverage as the control this
has to beat — and one trap in building it. That grid's "correct" cell is an **oracle**
(`OracleOutcomeContinuous`, `OracleTreatment`), which makes the good nuisance exactly right,
`R₂` exactly zero, and `TMLE`'s interval already valid; a study built that way would show
nothing to buy. The gap opens only where the good nuisance is *estimated*, so the study needs
a correctly-specified learner in that slot rather than the truth. Nightly tier; never run it
in the sandbox.

**Built, run, and it found nothing to buy** — which is the most important result on this page
and is a negative one. A pilot at `n = 500` over 24 replicates with `glm` correctly specified
for `Q̄` and misspecified for `g` put `TMLE` and `DRTMLE` at coverage 0.958 apiece, biases of
−0.013 and −0.008 against a Monte Carlo standard error of 0.018; the mirror cell put both at
1.000. The trap above was avoided and the gap still did not open, because the *diagnosis* was
incomplete: a correctly specified **parametric** nuisance converges at `n^(−1/2)`, so the
product condition is nowhere near binding and `R₂` is a small constant times `n^(−1/2)` rather
than a first-order term. What this variant is for is an **adaptive** good nuisance converging
more slowly than `n^(−1/4)` — a Super Learner in enough dimensions — at an `n` large enough for
the decay to show. That is out of reach on a nightly budget rather than uninteresting: the
pilot's two `DRTMLE` studies took 358s and 372s against the plain estimator's 5s and 3s.
`tests/e2e/test_coverage_slow.py`'s `TestDoublyRobustInference` therefore guards what it can
— that the point estimate is still doubly robust, that the interval does not *cost* coverage,
that the standard error matches the spread of the estimates — and says in its own docstring
that it is not a demonstration. **So this variant currently rests on its derivation and on
`drtmle`'s implementation, with no end-to-end evidence that it delivers.**

### The one thing still to pin — settled, on an argument rather than a measurement

Of the three things this section said to pin before any code, two are
[settled above](#what-the-source-settles) — which equations there are and which nuisance each
fluctuates, and which influence curve is reported. The third was **how the reduced regressions
are cross-fitted**, and it is settled now, in `fit_reduced`'s docstring where it belongs.

The problem is real and is stated correctly: their *design* is itself an out-of-fold
prediction — `ĝ(W)` or `Q̄(a, W)` — so fitting them on the same `Folds` trains fold `k`'s
regression on design values produced by models that saw fold `k`. Row `i`'s own data reaches
its own prediction through the *other* rows' design values, which is the dependence
`tests/unit/test_crossfit_leakage.py` exists to prevent, arriving through the design matrix
rather than through the target. `drtmle` cross-validates the reduced regressions alongside
every other nuisance and does not distinguish this case, and the software paper does not
discuss it, so there was nothing to transcribe and nothing to defer to.

**But the choice this section named is not a choice.** It offered "reuse `nuisance.folds` or
draw an independent split", and an independent split removes *none* of the dependence: the
contamination is in the design values, so which rows are trained on cannot undo it, and a
second split only loses the alignment with the fits it is a reduction of. The one construction
that does remove it is **per-fold designs** — predict `ĝ^(−k)` at every row, so fold `k`'s
reduced regression only ever sees designs from the model that excluded fold `k`. That costs no
extra fits, since `cross_fit_predictions` already builds that model and keeps only its
test-fold slice.

What it costs is worse than what it buys, and that is the decision. The training designs would
be that model's *in-sample* predictions and the test design its out-of-sample one, and a
reduced regression is a regression **of** the design — so it trades a second-order dependence
for a first-order covariate shift. So the split is reused, which is also what `drtmle` does,
and `groups` is forwarded so that the claim `test_crossfit_leakage` actually states — a model
must not train on rows standing in for the ones it predicts — holds at the level it is stated.

It keeps its own heading, and its shape, because a section that quietly absorbed an answered
question would read as though nothing had been open. What is worth carrying forward is that
this one was settled by *restating* it rather than by measuring it: the measurement this
section asked for would have compared two constructions of which one was never on the table.

### Scope, declared at what the derivation covers

**Binary, `mean` group** — `ey1`, `ey0`, `ate` — and, in the event, **one** reduction rather
than both: `reduction="univariate"` is written and `"bivariate"` is refused by name. Both are
derived in the sources and both were in scope here; the second was cut because it is a
different extra equation on a two-column design rather than a wider loop over the first, and
nothing was waiting on it. A `guard=` keyword says which of the
extra equations are solved at all, `drtmle`'s vocabulary for the same choice, and an empty one
is a plain TMLE. Both reductions are in scope because both are *derived* in the sources: the
bivariate one is van der Laan (2014)'s and the univariate one is Benkeser et al. (2017)'s
replacement for it, and the software paper states both sets of equations.

**A multi-valued treatment is not in scope, and the reason is worth writing down**, because
the obvious reading of the source says otherwise. `drtmle(a0 = c(0, 1, 2))` reports
treatment-specific means at `K` arms and the software paper works an example; the estimating
equations are written with a free `a` and nothing in them has a two-arm step. What is missing
is the derivation: van der Laan (2014) states its problem for "a subsequently assigned
**binary** treatment", and no theorem read here covers `K` arms. An implementation that
accepts an argument is not a proof that the argument is licensed, and the gap is not
hypothetical — the per-arm mechanism tilts do not renormalise, so the targeted `g*(·|W)` at
`K` arms is not a distribution over the arms, and whether that is harmless is exactly the sort
of thing a theorem would say and an example would not. It stays a candidate rather than a
refusal-on-principle: what would settle it is reading the multi-arm case in the 2017 paper, and
if it is there, the widening is a wider loop plus a multi-arm mechanism tilt — which
`solve_mechanism` does not have, since `ipsi` declares `requires_binary_treatment` and has
never needed one.

Every other axis this package has (`att`/`atc`, `regime`, `shift`, `ipsi`, `msm`) must be
**refused by name** rather than silently handed a plain fluctuation, on the rule `LTMLE`
established: a subsystem that was never taught about a variant raising `AttributeError` is not
a refusal. Each is a different score equation with no reduced-regression derivation behind it —
a stronger reason than the multi-arm one, which is a gap in what has been read rather than in
what exists.

Two subsystems still need deciding rather than inheriting, and one of them has moved.
`sensitivity/omitted_variable.py`'s Riesz representer reads the clever covariate's columns by
the arm each targets (`submodel.column_for`, since item 6) — [seam
6](#six-seams-where-the-sizing-names-two) is why that survives, and it survives because of a
default in `drtmle` rather than because of anything decided here, so it is an argument to write
down rather than a coincidence to rely on. And the truncation curve and the MNAR tilt reach the
targeting through `retarget`, so they would re-solve the extra equations — probably right, and
right by inheritance rather than by decision, which is how the wrong version of it would also
arrive. Seam 5 is the sharper half of that same question.

**Combining it with `CTMLE` is a derivation rather than a composition**, and the seams make
that easy to miss: both variants override `_nuisances`, both are `mean`-only, and a
subclass that ran the selection and then fitted the reduced regressions against the chosen `ĝ`
would run. (`CTMLE` is binary and this is not, which narrows where they overlap without
removing it: a combined fit would be binary, and the argument below is unaffected.) Two things
stop it. A reduced regression conditions on the *other* nuisance's
estimate, so it reads `ĝ` as a covariate — and `CTMLE`'s `ĝ` is deliberately not an estimate
of `g₀`, the collaborative point being that `g` need only adjust for what `Q̄` missed. And
`CTMLE` scores its path by the cross-validated loss of the *targeted* `Q̄`, so the criterion
choosing `ĝ` presupposes that `Q̄` is informative — which is precisely the cell this variant is
insuring against. The cost is the visible half of the problem and not the important one:
`cross_validate` rebuilds the path inside every selection fold, so each position would carry
its own set of reduced regressions and its own alternation. Refuse it by name, beside the
`incremental=` refusal it is a cousin of — there each candidate `ĝ` defines a different
`Ψ(δ)`; here it defines different reduced regressions.

### Sizing

Comparable to `CTMLE` or larger, and **not** transcription in the way the five items below
were. `CTMLE` swaps one array and inherits every influence curve, sensitivity analysis and
diagnostic untouched; this adds arrays, a targeting branch, curve terms and a serializer
version, and each of those is a place a reader could be told a plain TMLE's number. Four to
six commits in `src/`, plus a remainder module, a nightly coverage study and a section of the
guide — and the remainder module first.

Reading the source moved that upward a little, and the increase is nameable rather than a
hedge: three reduced regressions rather than two, two reductions rather than one, and a
`retarget` contract to decide in writing rather than to inherit. Call it five to seven, in the
order the seams are numbered and with the remainder module still first: remainder module, then
the reduced regressions, then the alternation, then the curve, then the serializer.

**It came out at four commits in `src/`, and the order was different twice.** The remainder
module went first as planned. The reduced regressions followed — and took the serializer bump
with them, off the end of the list, because `NuisanceEstimates` is reconstructed field by
field on load and a field added without one reloads silently as `None`. A version bump is
cheap to write while the reason for it is in hand and expensive to remember afterwards, so it
belongs with the array rather than with the estimator that reads it. Then the covariates, the
alternation, and **the estimator and the curve together** — the second departure, and the
sharper one. The list has them apart; landing them apart would have left `DRTMLE` reporting a
plain TMLE's interval under a doubly-robust name for the length of a commit, which is the
failure mode the whole section is organised around.

So the sizing was low by about one commit and right about the shape. What it got wrong is
elsewhere: it treated the reduced regressions as the hard part and the alternation as
transcription, and the alternation is where every surprise was — a stall rule that had to
watch the objective, three scores that go stale in three different ways, and equation (10)'s
conditioning.

**Theorem 1 of Benkeser et al. (2017) was a prerequisite of the curve commit and is not
met.** It is the only thing on this page that has to happen outside the repository, and it did
not: Biometrika is paywalled and this environment's network policy denies the working-paper
mirrors. The curve landed anyway, labelled where a reader meets it — in
`reduced_corrections`' docstring, in the methodology section and in the guide — as **what
`drtmle` computes rather than what the theorem derives**. That is a deliberate, stated
exception to this page's own rule, not an oversight, and it is the first thing to close.
There is a cheaper check that was also not done and would catch much of the same class of
error: **no number here has ever been compared against `drtmle`'s own output.** The package
has no cross-language test at all, so this is not a new gap; it is a more costly one here than
elsewhere, because the whole variant is a variance estimate transcribed from that
implementation.

The thing that did *not* move is worth saying, because the temptation runs the other way. The
source's implementation accepts more than its derivation covers — a multi-valued treatment
most visibly — and none of that is in the sizing. What an implementation accepts is a fact
about the implementation; the scope above is set by what has been derived and read, and it is
allowed to be narrower than either paper's software.

Three of `drtmle`'s own settings are worth recording as the contrast this section is for,
because each is a place where transcribing the implementation would import a decision this
package has already made differently. Its defaults are `maxIter = 3`, a mechanism truncation of
`1e-2` and a score tolerance of `1/n`, against `max_outer = 50`, `g_bounds="auto"` and
`tol = 1e-10` here. And its mechanism fluctuation **silently sets a divergent coefficient to
zero**, where `_newton_logistic` reports a `TargetingFailure` — the difference between a fit
that quietly declines to target and one that says it could not.

The first of those three was very nearly imported anyway, and the near-miss is the most useful
thing on this page. A fit whose fold split was drawn unseeded ran to the outer cap with the
two extra scores at `1e-5` and `score_check` reporting NO, and the response drafted was to
hold a doubly-robust fit to a *statistical* tolerance — `se/√n` rather than `1e-3·se/√n` — on
the argument that these equations cannot be solved to machine precision and `o(n^{-1/2})` is
all asymptotic linearity asks. The argument is sound and the change was still wrong, for a
reason that took one measurement: across six **seeded** fits at `n = 800` the alternation
converged in 15 to 45 rounds with a worst score of `1e-9`, and the ordinary tolerance passed
every time. The failure was a minority draw, not a property of the estimator, and blunting the
diagnostic a thousandfold for every doubly-robust fit would have hidden the next real one.
**A threshold changed after seeing a failure needs the failure characterised first**, and the
order those two happened in is the whole lesson. What survived from the episode is the defect
it turned up on the way: the reported score and the reported curve were being read off
*different* refits of the reduced regressions, which is why the per-estimand row disagreed
with the per-equation rows by two orders of magnitude.

## Refusals worth lifting

Everything under [Not written yet](methodology.md#not-written-yet) is a candidate; these are
the ones that answer a question applied causal inference actually asks *and* rest on a derivation
that is already settled, so the work is transcription and checking rather than research.
Nothing here is blocked on a modelling question.

**The order is a dependency order, not a preference order.** Each item is independently
shippable, but taken in sequence some of them hand work to the next: the first was
self-contained and unblocked the sixth, whose whole content is the contrast machinery it
built; the second builds the projection
machinery the fourth copies; the third and fourth both change `fit_regimen` and
`fit_mechanism`, so doing them adjacent is one round of churn in those signatures rather than
two — and taking them adjacent paid: the third left the recursion carrying the data's
weights, which the fourth inherited rather than adding.
The fifth was last because its cost is dominated by test infrastructure rather than by
derivation — it is the only one needing a *new* oracle law rather than a branch on an
existing one, and that held: the `src/` change was four small commits and the law, its
Gateaux module, its remainder module and the mutation hunting were the rest of it.
The sixth is the only one that touches no estimand at all: it lifts two refusals *around*
the fit rather than adding a parameter to it, which is why it needed no oracle branch and
no registry entry, and why the [oracle-law gate](methodology.md#the-oracle-law-gate) has
nothing to say about it.

**All six have landed**, and so has the second variant above. What remains is not an item:
Theorem 1, a cross-check against `drtmle`'s own numbers, and a coverage study that finds the
gap the variant exists to close — all three set out under [What `drtmle`
touched](#what-drtmle-touched) — plus a handful of refusals under [Not written
yet](methodology.md#not-written-yet) that are there because nobody has asked rather than
because anything stands in the way.

1. **`ATT` / `ATC` for a multi-valued treatment — landed.** "The effect among those who
   actually received arm `a`" is now `att[a vs ref]`, one per non-reference arm, with
   `atc[a vs ref]` the same contrasts among the reference arm's units; see [multi-valued
   treatment](user-guide.md#multi-valued-treatment). The derivation was the binary one with
   `1{A=1}` and `1{A=0}` replaced by `1{A=a}` and `1{A=r}` and the odds by `g_a / g_r`, so the
   fluctuation gained a column per contrast rather than a group, and
   `tests/discrete_law_multi.py` gained `att[a vs r]` branches rather than a new law. Two
   things worth recording. The reference arm loads *every* column of that fluctuation —
   it is the arm each contrast is taken against — so the Hessian is no longer diagonal as
   the `mean` group's is, and `Submodel.contrast_columns` exists because a column is now
   keyed by the contrast it carries rather than by an arm it updates. And they are **not**
   in a multi-arm default report: `2(K-1)` further parameters would have moved the
   simultaneous bands of every multi-arm fit that already ran, so `default_arms="binary"`
   keeps them opt-in. What still followed from the same contrast machinery, and was the
   next thing this unblocked rather than part of it, is item 6: the omitted-variable bound
   and the MNAR tilt on a multi-valued treatment
2. **A non-identity link for `msm=` — landed.** `link="log"` and `link="logit"` make a
   coefficient a log risk ratio or a log odds ratio, and `res.coefficients(scale="ratio")`
   exponentiates them; see
   [the MSM section](user-guide.md#summarising-the-arms-a-marginal-structural-model).
   The identity path is bit for bit unchanged — `dm/dη` is one there, so the covariate is
   the same array and the projection the same `np.linalg.solve`. Three things worth
   recording, because the sizing above got two of them wrong. The alternation is **not**
   `solve_with_mechanism`: that loop terminates because its two steps are coordinate ascent
   on one joint likelihood, and a projection is a least-squares solve rather than a
   likelihood, so `solve_with_projection` is a sibling that restarts from `Q̄⁰` each round
   for a clean fixed point instead of continuing for a monotone one. It converges much
   faster than the mechanism loop besides — `1e-3` to `1e-4` per round against 0.15 to 0.52
   — because `β` reaches the covariate only through a smooth factor. The matrix the curve
   is premultiplied by the inverse of gains a **curvature term** that vanishes only where
   the working model fits, so no saturated check can catch its absence; that mutation is
   now a control in the oracle. And the remainder stops being *exactly* zero when the
   mechanism is right — that exactness was the linearity of the estimating equation in `β`,
   not a stronger double robustness — so the test measures a rate where it asserted an
   equality. What this unblocks is the fourth item, which copies the projection's shape
3. **Observation weights for `LTMLE` — landed.** `LTMLE(...).fit(frame, weights="w")`
   estimates the declared regimen parameters in the tilted population `dP_w = w dP / E[w]`,
   with every node's mechanism, censoring factor and sequential regression fitted by
   weighted loss, every node's score equation weighted, and the reported curve
   `(w / E[w]) · D*(P_w)`; see [treatment over
   time](user-guide.md#treatment-given-over-time). It was
   the transcription the sizing said it was — the statement was already derived in
   `data/weighting.py` and the plumbing already carried a weight vector — but three things
   are worth recording. The **refusal's stated reason was wrong**, in the same way item 5's
   was: it said observation weights "put a further per-unit factor in the clever covariate's
   denominator at every node", and they do not. A weight tilts the *population*; `h_t`
   divides by the `2T` mechanism factors and by nothing else, and putting `w` there would
   divide the estimating equation by the very tilt it applies. What *does* move is
   `g_bounds="auto"`, resolved at Kish's effective `n` as it is at one node — and over `T`
   nodes that compounds rather than cancels, since the bound reaches every factor. And the
   leverage `res.diagnostics()` reports is now `w / ∏g` rather than `1/∏g`, on
   `sensitivity/positivity.py`'s reasoning that the two reweightings multiply. The oracle
   was a branch on the existing law rather than a new one: `tests/discrete_law_longitudinal.py`
   gained the tilt and its Gateaux derivative, and its saturated learner had to start
   *honouring* `sample_weight` — accepting and discarding it would have left the estimator
   holding `P_0`'s conditionals while its estimand was at `P_w`, which is the one mistake
   here that leaves `epsilon` non-zero rather than silent. The nightly tier gained a
   coverage study on a *biased sample*, where selection is a known `π(W_1)` and `w = 1/π`,
   so the truth is the unweighted one unchanged; ignoring the weights there costs about
   fourteen Monte Carlo standard errors of bias on each counterfactual mean — and almost
   nothing on their contrast, which is why that control is taken on a level
4. **A marginal structural model over regimens, for `LTMLE` — landed.**
   `LTMLE(regimens, msm=MSM(...))` reports `msm_regimen[<term>]` in place of a mean per
   plan: `β` is the `h`-weighted projection of `E[Y^ā | V]` onto `m(ā, V; β)`, under every
   link, with `V` a subset of the baseline covariates and the horizon *inside* the design
   on a survival fit — see [summarising the
   regimens](user-guide.md#summarising-the-regimens-a-marginal-structural-model). Four things
   are worth recording, and the sizing above got two of them wrong.
   The structural difference from the point-treatment working model is that **the node
   fluctuation must be pooled across the regimens**, and the reason is a rank argument
   nobody had made: there the `p` columns are separated by summing over the arms *within a
   row*, and a regimen is a plan rather than a value some unit took, so a per-regimen
   covariate is `φ(ā, V)` scaled by the scalar `h_t` — rank one whenever the model has no
   effect modifier, collapsing its `p` score equations into one. Each node therefore solves
   one fluctuation over the regimens stacked, with a single shared `epsilon`, and the real
   churn was control flow rather than mathematics: `fit_regimen`'s one-plan-at-a-time
   backward pass had to become lockstep.
   **A link costs a whole backward pass per round**, not a re-solved fluctuation, which the
   sizing had not seen: `β` enters every *earlier* node's regression target through the
   recursion, so there is no fixed `Q̄⁰` to restart from and the fixed point is stated over
   the whole pass. Measured at four or five rounds, contracting by `1e-4` each — the
   point-treatment rate, and for the same reason.
   `solve_projection` was reused **verbatim** with its arm axis read as the regimen axis,
   which is what item 2 bought; `MSMSet` deliberately was not, since its second axis is
   arms in its field name, its docstring and its accessors, and its constructor reads a
   `CausalData` throughout. Only the rank rule is shared. `h(ā, V)` and the observation
   weights stayed apart exactly as item 3 predicted.
   Two claims turned out weaker than expected and both are now stated as they are. The
   saturated reduction is **not** bit-for-bit and cannot be — the pooled Newton's
   convergence test and line search are taken over all the stacked rows — so it is exact on
   the exact law, where no step is taken, and `1e-11` elsewhere. And of seven deliberate
   mutations, **three passed on the first try**: the baseline-frame pin was blind because
   every longitudinal fixture here has an empty `time_varying[0]`, making
   `history_frame(1)` the same object; dropping the observation weight from the curve was
   invisible because nothing exercised a working model on a weighted fit; and the source's
   claim that an `at_risk` mask "leaves `epsilon` non-zero" was simply false, since the
   covariate is already zeroed off `trained_on` and the substitution moves no reported
   number at all. The first two are now covered and the third is now stated correctly
5. **`shifts=` with `delta=` — landed, and with `intermediate=` and `weights=` besides.**
   `_refuse_continuous_combinations` refused all three on one reason and the reason was
   wrong for all three, so lifting them was one change rather than three; see [missing
   outcomes, an intermediate, and weights on a
   dose](user-guide.md#missing-outcomes-an-intermediate-and-weights-on-a-dose). The
   derivation was the existing one with a further factor, exactly as it had been for
   `incremental=` with `delta=`: `H(a, W) = h(a, W) / {π(a, W) q_z(a, W)}`, and only the
   residual term is inverse-weighted because `Q̄(d(A,W),W) − Ψ` is a function of `(A, W)` and
   both are recorded whatever happens to `Y`. Five things are worth recording, and the sizing
   above got three of them wrong.
   The `(n, S + 1)` array belongs on **`NuisanceEstimates`, not on `ShiftSet`**, which the
   sizing had backwards. `bounded_missingness(nuisance_bound)` truncates at *targeting* time,
   and `retarget`, the MNAR override and `truncation_curve(mechanism=True)` all depend on
   that; folding `1/π` into `ShiftSet.design` at fit time would freeze the bound and make the
   mechanism truncation curve **flat by construction** — which reads as "the estimate does not
   hinge on the truncation choice", a wrong conclusion reported silently. Keeping it where the
   arm path keeps it also meant no `ShiftSet` field, no `subset` branch and no serializer
   change, and `clever_covariate_inputs` worked untouched.
   **A Gateaux check on an exact law cannot see the mistake this item is about.** At `epsilon
   = 0` the reported curve reads the *observed* block of the covariate and the untargeted
   `Q̄`, so dividing every block by the mechanism at the observed dose — the whole error the
   `(n, S + 1)` array exists to prevent — passes all 39 tests of the new Gateaux module. So
   does applying the selection indicator to the counterfactual blocks. Both were applied and
   watched to pass before the two instruments that *do* catch them were written: a structural
   pin on the covariate's blocks, and a plug-in with `epsilon != 0`. This is item 4's lesson
   again and it did not announce itself the second time either.
   `intermediate=` came free and turned up a live bug on the way. `mtp_submodel` applied the
   `1{Z = z}` indicator to its counterfactual blocks, where `mean_submodel` deliberately does
   not — the blocks are already at `Z = z` by construction — so every row whose intermediate
   took the other level would have carried an **un-updated** prediction into the plug-in. Dead
   code while `intermediate=` was refused here; a silent bug the moment it was not.
   `weights=` needed no `src/` change at all beyond deleting the refusal: `fit_conditional_density`
   already routed the weights through the long expansion, `shift_means` already averaged and
   scaled by them, and `_bounds_n` already resolved `auto` at the effective `n`. Item 3's
   compounding-bound story does **not** carry over, though, and saying so matters: `g_bounds`
   does not bite on this axis at all, since there is no per-arm propensity and the ratio is
   untruncated, so `nuisance_bound` is the only truncation a shift fit has.
   The oracle law was the cost the sizing said it was, and one law rather than the predicted
   one-per-lift: `tests/discrete_law_shift_cde.py` crosses the doses with `(Δ, Z)` and takes
   `level=None` for the parameter a `delta=`-only fit reports, so the two cannot disagree by
   construction. Its one indispensable property is that `π` and `q_z` vary with the **dose** —
   a mechanism depending on `W` alone makes `π(d(a,w), w) = π(a, w)` identically and the whole
   feature untestable. With the law's own nuisances the fit returns the truth *exactly* at all
   three levels; the mechanism's quantile binning is why that, and not a coverage study, is
   the strongest end-to-end statement available here.
   What is left refused is a narrower gap than the one it replaced: the **MNAR tilt** on a
   shift fit. The tilt re-mixes `Q̄` under a moved mechanism, a shift's plug-in is `Q̄` at the
   assigned dose, and whether the tilted parameter is still the shift parameter has not been
   derived — so this is a missing derivation rather than missing transcription, which is why
   it is not being carried forward as an item of its own
6. **The omitted-variable bound and the MNAR tilt on a multi-valued treatment — landed**,
   and the E-value with them. `omitted_variable("ate[medium vs low]")`, `robustness_value`,
   `benchmark`, `contour`, `evalue("rr[medium vs low]")` and `missingness_tilt()` are now
   one analysis **per contrast**; see [multi-valued
   treatment](user-guide.md#multi-valued-treatment). This is what item 1 unblocked, and
   its sizing — one sentence rather than a section — got the size right and one of the two
   reasons wrong. Five things are worth recording.
   **The bound's refusal reason was wrong**, in the way items 3 and 5 found theirs to be
   and for the third time in six items. It said the bound "rests on a scalar confounding
   strength in the treatment equation, and with more than two arms an omitted covariate has
   one such strength per arm — a different derivation, not a wider loop". But `cf_d` is not
   a coefficient in a treatment equation: it is the share of the *Riesz representer's*
   second moment a confounder would add, and a representer belongs to one linear functional.
   `ate[high vs low]` has one and `att[mid vs low]` has another, so it is exactly a wider
   loop — one bound per contrast, each with its own `ν²`, and none of them a summary of the
   others.
   **The tilt's refusal named a real choice** — "whether gamma is shared across arms or per
   arm" — and the answer is a *direction* rather than a scalar or a vector grid. Shared
   `gamma` stays the default, since that is what the two-armed path always did;
   `arm_gamma=` declares one multiplier per arm and the grid sweeps its magnitude, so
   Scharfstein–Rotnitzky–Robins's per-arm parameter vector is reachable (`arm_gamma=v` with
   `gamma=[1.0]`) while `tipping_gamma` stays a one-dimensional root find, which is what
   makes a tipping point a single number. Every arm must be named, and the returned frame
   carries a `gamma[<level>]` column per arm: an arm defaulted to 1 — or a direction living
   only in the call — would be the quiet choice the keyword exists to make loud.
   **A default multi-arm fit reports only `ey[...]`**, so the tiltable set had to include
   the per-arm means or the lift would have delivered a tilt with nothing to tilt. The old
   filter was the literal tuple `("ate", "att", "atc", "ey1", "ey0")`, and replacing it with
   a name → arms map *composed forward* through `parameter_name` — `sensitivity/_parameters.py`,
   shared by all three analyses — is what made that visible rather than a later bug report.
   **Two latent binary bugs fell out, both about the declared reference**, and no existing
   test could see either because two arms and `reference=0` are where the constants they
   replaced were right by default. With `reference=1` the tilt reported `E[Y¹] − E[Y⁰]`
   under the name of an `ate` the fit had defined the other way round, and weighted the ATT
   by `A` where the parameter conditions on `A = 0`; and the E-value's risk-difference
   conversion divided by `ey0` where the baseline it needed was `ey1`. Reading the arms off
   the parameter fixes both, and they are pinned by a two-armed fit that declares the other
   reference — `TestTheTiltFollowsTheDeclaredReference` fails on all three estimands if the
   first comes back, and `test_two_arms_divide_by_the_declared_reference_too` on the second.
   **The instruments differ because the two objects do**, and that is the part worth
   carrying forward. The bound is a closed-form functional of the nuisances, so
   `tests/discrete_law_multi.py` with its own nuisances makes `σ²` and `ν²` exact functions
   of the eighteen cell probabilities — written out longhand, at `1e-12`, for all nine
   parameters and *both* estimators of `ν²`, which must agree there by the Riesz identity
   and so check the arm indexing in a way neither alone could. The tilt is not a functional
   of the law at all but a re-mixing of the *fitted* regression, so no exact law has
   anything to say about it: its instruments are that `gamma = 0` reproduces the whole
   report and that an arm with `π ≡ 1` is left exactly where it was however the others are
   tilted. No new law, and eleven mutations watched to fail

## On native acceleration

A Rust extension for the numerical kernels was planned. `benchmarks/bench_tmle.py` says it
is not worth building. Profiling a full fit by module (`cProfile`, total time):

| fit | cleverly-authored code | scikit-learn + LightGBM |
| --- | --- | --- |
| n=5,000, `library="default"` | **0.5%** | 44% |
| n=20,000, `library="glm"` | 22% | 17% |

The targeting step is 1.5–1.7% of a `glm` fit and does not appear at all in a `default`
one — it is a 2×2 Newton solve with a closed-form Hessian. Nuisance estimation dominates,
and it already runs in compiled code. Note how much the preset matters: `glm` is the
cheapest library available, so it makes every other line's share look several times larger
than it is. Benchmark with `--library default` before drawing a conclusion.

The 22% figure above is almost entirely *one* function, and profiling it turned up waste
rather than arithmetic — waste that was cheaper to fix than to rewrite:

- **The multiplier bootstrap was 92–95% multiplier *generation* and 2–3% matrix product.**
  It drew a full float64 uniform to produce one Rademacher sign. Generating bits instead
  is ~2.4× faster. Better: for `multiplier_kind="normal"` the max-t law has a closed form
  — `xi @ IC` is a linear map of a Gaussian — so the whole resampling loop collapses to
  one covariance and a draw from an *m*-dimensional normal, which is **80–360× faster**
  and never allocates a `(n_replicates, n)` array.

  That speed is not free, and `multiplier_kind` still defaults to `"rademacher"`. The
  closed form exists *because* the Gaussian max-t law depends on the influence curves only
  through their covariance — so `"normal"` is a plug-in normal approximation rather than a
  resampling scheme, and it cannot see the leverage a `1/g(W)` clever covariate produces
  under weak overlap. Simulated against a brute-force max-t distribution, it is biased
  conservative there (+0.14 on a true 2.16 at n=200, +0.07 at n=2,000), while `"rademacher"`
  stays within 0.02. On well-behaved influence curves all three kinds agree. Use `"normal"`
  when *n* is large, the curves are well behaved, and resampling actually shows up in a
  profile.
- **The cluster bootstrap rebuilt its membership index inside every replicate**, an
  `O(n_clusters × n)` scan per draw. Building it once is **24–160× cheaper** per replicate,
  which a 1000-replicate cluster bootstrap pays back a thousand times over.
- `cluster_sums` used `np.add.at`, which is unbuffered; `np.bincount` is ~2× faster.

None of that needed Rust, and the package stays pure-Python. The other place that mattered
turned out to be thread scheduling rather than arithmetic: nuisance fits run
single-threaded by default so parallelism happens across folds and candidates instead of
inside each fit (see `cleverly.learners.set_thread_limit`).

**When to revisit this.** Native code pays where the nuisance estimator is *not* an
scikit-learn model, and today none of them is. The trigger is **HAL** (highly adaptive
lasso) and its undersmoothed variant: a zero-order spline basis of `n × O(n·d)` binary
indicators that scikit-learn's lasso cannot take, where basis enumeration, sparse assembly
and coordinate descent are a natural fit for a native extension — R's `hal9001` ships a C++
backend for exactly this. The EP-learner benefits *through* HAL rather than on its own; its
other cost is targeting a *k*-dimensional score with *k* = basis size, which is BLAS-bound
and already fine. Longitudinal and survival TMLE are weaker cases: the loop over timepoints
is Python, but each body is a nuisance fit, so they stay scikit-learn-bound. That remains a
prediction rather than a measurement — `benchmarks/bench_tmle.py` has no `LTMLE` case, so
profile one before acting on it.

The measurement is reproducible — rerun the benchmark before revisiting this.
