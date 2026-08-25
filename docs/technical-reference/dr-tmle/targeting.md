# Targeting and cross-fitting, as implemented

The theorem fixes the estimating equations. It does not fix the order they are solved in, the
number of rounds, the fold layout, or where the truncation bounds sit. This page records those
choices and the evidence for each. [The theorem-backed contract](theorem.md) carries what the
theorem itself states.

## The alternation

The paper's recursive algorithm uses `H_1(g) = a/g`, `H_2(g_1,g_2) = a·g_2/g_1` and
`H_3(Q_r,g) = Q_r/g`. It runs in six steps.

1. Initialise.
2. Fluctuate `Q̄` along `H_1`.
3. Estimate `g_{r,1}` and `g_{r,2}` at the once-updated outcome regression.
4. Fluctuate along `H_2`.
5. Estimate `Q_r` at the twice-updated one, then fluctuate `g` along `H_3`.
6. Iterate until the three empirical means are approximately zero.
The outcome fluctuations are fitted **using only rows with `A = a`**, which is this package's
per-arm indicator design.

Two things the algorithm has no counterpart for and this implementation adds: a **closing pass** at
frozen final reductions, so the reported scores are the scores of the state returned; and the
truncation of [the previous section](targeting.md#the-bound-inactive-scope).

The alternation is **not guaranteed to converge.** Equation (10)'s covariate becomes small on
exactly the fits anybody wants, so its inner solve can be singular or stop at working precision.
The archived 96-fit sweep had 87 tolerance exits, 8 stalls, and 1 cap. It had cross-fitting disabled
and therefore did not cover the shipped 10-fold default.

A fixed-seed `glm` check of the current source produced the following default-path evidence. The
counter's historical name is `ill_conditioned`, but it counts every equation-(10) inner failure,
including a tolerance-limited full-rank solve; `res.validate()` therefore describes these as
numerically difficult rather than asserting that every Hessian was singular.

| n | primary folds | exit | numerically difficult rounds | final score check |
| ---: | ---: | --- | ---: | --- |
| 200 | off | cap | 0 of 50 | passed |
| 200 | 10 | stall | 3 of 46 | passed |
| 1000 | off | tolerance | 0 of 44 | passed |
| 1000 | 10 | tolerance | 2 of 16 | passed |
| 3000 | off | cap | 0 of 50 | passed |
| 3000 | 10 | tolerance | 0 of 8 | passed |

The affected 10-fold `n=200` solves had full-rank two-column designs and absolute scores between
`4e-13` and `9e-13`; the statistical score check passed because those residuals were negligible.
No argument here *proves* the iterates approach a common zero of the three equations, which is why
[the diagnostics](diagnostics.md) surface both convergence and numerical difficulty.

**Every one of those verdicts is read on two rulers, and it has to be.** Equations (9) and (10)
carry covariates that vanish where their own nuisance is right. A score *relative to the
covariate's own magnitude* is then a ratio of two small numbers. It cannot reach a tight bar
however well the equation is solved. Each equation therefore counts as solved when it clears
either the relative tolerance or an absolute bar of `1e-3 / n`.

The loop's exit test always used both. Two verdicts read off the *result* did not, and the
registered study is what surfaced it. Of 24 recorded solver failures, 9 had an absolute worst
score below both that bar and the one `score_check` applies to the reported fit. `converged` was
also false on fits that exited cleanly at tolerance. `failure` and `converged` now use the same
pair.
Correcting them is bit for bit: over 360 estimates on the paper law at `n = 3000` the largest
change was `9.7e-17`, because both were always statements about the diagnostics rather than about
the fit.

Two neighbouring things deliberately keep the relative bar alone.

`closing_capped` records how the closing stage *ended*, not whether the score it left matters. It
is routinely true on a fit that solved everything, because equation (9)'s covariate reads the
mechanism it tilts. Read it beside `failure`, which answers the other question on both rulers.

The closing stage itself exits on the relative bar because of what it is for. It exists so the
mean of the curve the fit *reports* is zero, it costs arithmetic rather than refits, and running
every step is therefore its cheap outcome. Letting it stop on the absolute bar moved the reported
curve's mean from `5.8e-7` to `1.5e-6` at `n = 600`, which
`tests/unit/test_drtmle_fit.py::TestTheCurveReadsWhatTheAlternationLeft` caught.

## The update order

`update_order="drtmle"` (default) or `"benkeser"`. **A diagnostic keyword rather than a tuning
one.** `"drtmle"` follows the canonical R package: equation (9), refit only `g_{r,1}` and
`g_{r,2}`, equations (10) and (8), then refit only `Q_r`. `"benkeser"` follows the published
six-step recursion. The working paper's step 7 states its termination as the three empirical means being
approximately zero, so its six-step order is one route to a fixed point rather than something
Theorem 1 assumes about the returned collection. The theorem's hypotheses are conditions on the
returned collection, not on the route. `"benkeser"` implements that order beside R `drtmle`'s,
sharing the stopping rule, stall test and closing pass, deliberately: what is in question is the
route, and a comparison in which two things differ answers nothing.

Two cautions apply. Compare the **scores and the estimates**, not the fluctuation coefficients. The
submodels a round passes through differ, so an `epsilon` from one is not an `epsilon` from the
other. And compare at the **same nuisances**: same data, same `random_state`.

**How far the two routes agree depends on which nuisance is wrong.** An earlier reading of one
draw at `n = 600` recorded that they agree on `ψ` and differ on `σ²_n`, and generalised from it.
The registered study measures the same comparison on the paper law at `n = 3000` over 120 paired
draws, and the generalisation does not hold.

| regime | route difference in `ate`, as a share of one sampling standard deviation | worst draw |
| --- | ---: | ---: |
| both nuisances correct | 5.4% | 19% |
| outcome regression correct | 5.4% | 27% |
| treatment mechanism correct | 40.6% | 162% |

Both routes solve all three equations to about `1e-11` on those draws, so this is not one route
failing to converge. **The three equations do not pin down a single answer when the outcome
regression is misspecified.** Step 7 states its termination as the three empirical means being
approximately zero, and on this law more than one state satisfies that: the route decides which
one is reached. For scale, the disagreement between this package and R `drtmle` in the same cell
is 47.6% of a sampling standard deviation, so *cleverly against cleverly* and *cleverly against
R* are the same size of difference and have the same cause.

The reported `σ²_n` differs for the reason it always did, and that part stands: the exit condition
constrains the three empirical *means*, while the reported variance is the second moment of a
curve built from the reductions, which the two routes leave at different vintages by construction.
The median `|Δσ|/σ` is 0.02% and 0.21% where the outcome regression is right, and 2.54% where it
is not. Both routes ran at the registered `max_outer = 100`.

Read `update_order` as a diagnostic, and read a large route difference as a statement about the
law rather than about the code. The
[canonical DR-TMLE study](../method-evidence/canonical-dr-tmle.md) carries the
measurement.

## How many rounds

`max_outer` (default 50) caps the alternation. It is **not** `max_iter`, which caps the Newton
steps inside one fluctuation, and the two were confused in this project's own published record:
the registered DR-TMLE manifest listed `max_iter: 100` as its alternation setting while the loop
ran at a hard-coded 50 that no caller could reach. The value that applied is now on
`result.repeats[0].fluctuations["mean"].reduction.max_outer`.

Raising the cap changes only the fits that reached it. Measured on the paper law at `n = 3000`
over 120 draws, going from 50 to 100 left every tolerance exit bit for bit and moved the 7 fits
that had hit the cap by up to `1.3e-3`, which is 7% of one sampling standard deviation. Act on a
`"cap"` exit rather than noting it: it says the draw had not settled, and the estimate reported
is the one the loop stopped at. Read it beside `score_check`, which says whether the scores the
fit left are small enough to matter.

## The canonical `cvFolds` mapping

R `drtmle`'s `cvFolds` path maps to `cross_fit=True, reduced_crossfit="pooled",
targeting_scheme="pooled", cv_evaluation=False`. The
[evidence manifest](../evidence.md#estimator-variants-over-registered-targets) states the map, the
test that pins it, and which settings the source audit refuses.

## Reduced-regression cross-fitting

`reduced_crossfit="pooled"` (default) reuses the primary split as it stands; `"nested"` takes
fold `k`'s training designs *and targets* from primary models fitted with fold `k` left out as
well. **A diagnostic keyword rather than a tuning one**, for a specific reason: the argument for
the cheap construction needs one quantity to vanish, and that quantity *is* the difference between
these two.

**Cross-fitting is not in the theorem.** The canonical mapping above is source evidence for the
implemented computation, and the 2023 article's statement that cross-validated nuisance estimates
weaken the entropy conditions is supporting evidence that cross-validated DR-TMLE is intended;
neither is the missing proof.
What is specifically unaddressed is the **pooled** construction, in which an observation influences
other rows' generated regressors and then returns to its own reduced-regression prediction through
those rows. Generic cross-fitting results do not cover that because the conditional independence
they turn on is exactly what the generated design breaks.

The argument supplied here. Writing `ĥ_k` for what fold `k`'s reduced regression contributes and
`h̃_k` for the nested version:

```text
(P_{n,k} − P_0) ĥ_k = (P_{n,k} − P_0) h̃_k             [A]
                    + (P_{n,k} − P_0) Δ_k ∘ ĝ^(−k)     [B],   Δ_k = Q̂_r^(−k) − Q̃_r^(−k)
```

**[A]** is the ordinary cross-fitting argument: conditional on the fold-`k` complement both maps
are fixed, so the term has conditional mean zero and is `o_p(n^(−1/2))` under the `L_2` convergence
Theorem 1 already assumes. **[B]** is the whole of the open question, because `Δ_k` is precisely
the object that depends on fold `k`; no cross-fitting lemma reaches it. What does is asymptotic
equicontinuity, needing two conditions:

> **(E)** the reduction learner's fitted functions of one or two scalars lie, with probability tending to
> one, in a class whose bracketing entropy is bounded **uniformly in the underlying measure**.

The structural fact that makes (E) available is that **the reductions have fixed dimension**:
one scalar for the default construction and two for the bivariate reduced probability, however
many covariates the fit adjusted for. Composition with a fixed map transports brackets exactly,
so the entropy requirement falls entirely on a class of functions of at most two variables and
not at all on the primary nuisances' complexity. The measure-free phrasing is load-bearing, since
the pushforward is a *random* measure and a bound holding at `P_0` says nothing about it. A
fixed-basis linear or logistic model is a bounded fixed-dimensional sieve; histogram
boosting is a fixed bounded-variation ball when its iteration and leaf limits are fixed. A
CV-selected round count would take boosting out of the class. A random forest is outside.

> **(S)** `‖Δ_k‖_{L_2} = o_p(1)`: the reduction fit is `L_2`-continuous in the design and target
> columns it is handed.

**(S) is the open condition.** It is free for a fixed-basis linear smoother and not free for
anything that *selects* structure from the data: a split point, a bandwidth, a CV-chosen
candidate. An arbitrarily small design perturbation can move any of those selections discretely.
Boosting is entropy-safe and design-continuity-unsafe. `reduced_crossfit="nested"` is what computes
`Δ_k`; a measured dispatch put its consequence on `ψ` at or below what a redrawn fold split moves
in every cell, which is **supported, not shown**, since a consequence can hold by cancellation.

Two things not to read into "nested". It costs `K` times the primary nuisance fitting, and was
measured at **1.3x to 17x** a pooled fit's wall clock over four draws, reaching the outer cap on
two. What dominates that cost is not the extra fitting: nested reductions are noisier, so equation
(10)'s near-singular solve takes more rounds. And **neither construction makes the targeted
collection fold-independent**: `epsilon` is solved on all `n` rows, since
`targeting_scheme="fold"` is refused, so a nested fit is *nested in the nuisance models and pooled
in the tilt*.

## Where the truncations are

On the univariate construction, `g_{r,2}`'s bound is fixed at **fit** time, because the array *is*
a regression of a quotient by the mechanism. It is the only bound in the package chosen at fit
time rather than at targeting time. Two consequences a reader will otherwise trip on.

- `DiagnosticsFacade.truncation_curve` moves the clever covariate's denominator and does **not**
  move these arrays. That part of the curve is flat *by construction*. `ReducedSet.g_bounds` is on
  record, so a reader of such a curve can find out that the sweep never reached them.
- `gr1` is stored **untruncated** and bounded at read time through `ReducedSet.bounded_gr1`,
  column by column and not complemented across arms.

One further condition sits beside (E) and is a rate rather than an entropy bound. `g_{r,2}`'s
target is `(1_a − ĝ)/ĝ` at the bounded mechanism, so its envelope is `1/lo − 1`, and equation
(10)'s covariate divides by `g_{r,1}` truncated at the same bound, so that envelope is `O(1/lo²)`.
Under `g_bounds="auto"`, `lo → 0` and the envelope **grows with `n`**. That pulls against the
bound-sequence row of [the scope section](targeting.md#the-bound-inactive-scope), where exactly that shrinkage
is what makes the truncation asymptotically inactive. Both are open.

## The bound-inactive scope

**Truncation is not in the theorem's algorithm.** The theorem's `D_g` is evaluated at the same
`g*` its score is solved at, and that `g*` is not truncated anywhere. One mechanism, produced by
an unconstrained `expit` fluctuation, appears identically in equation (9)'s covariate, in equation
(9)'s residual, and in `D_A`. Boundedness is an *assumption about `g_0`*, not an operation on
`ĝ`. So the theorem supports neither this package's original hybrid (bounded
denominator, raw residual) nor R's post-fit clip.

The package instead uses a **constrained estimating equation**. It puts `clip` inside `F` and
solves that for a root, so the final score *is* the declared estimator's score
(`fluctuation/mechanism.py::solve_bounded_mechanism`). Where nothing clips, that solver returns
the unconstrained solve untouched, so such a fit **is** the estimator Theorem 1 is stated for, not
an approximation to it.

The guarantee is therefore scoped rather than assumed through:

- a fit whose truncations are **inactive** is inside the theorem-backed contract;
- a **bound-active** fit is empirically supported and reported as outside it. It is not a failing
  fit: on `weak_overlap_dgp` every identity holds at `1e-17` and every score is negligible while a
  third of the `(row, arm)` pairs clip at the initial mechanism.

**Three truncations must be inactive, not one**. A randomized missing-outcome fit has five
truncations because it divides by two separately bounded mechanisms. The last truncation has no
corresponding theorem assumption:

| truncation | witness on the fit | in Theorem 1's assumptions |
| --- | --- | --- |
| `ĝ` at the fit | `CorrectionRow.initial_clipped` | an assumption on `g_0`, not an operation on `ĝ` |
| `π̂` at the fit for missing-outcome fits | `CorrectionRow.observation_clipped` | an assumption on `π_0`, on the same footing |
| `g*` at the exit | `CorrectionRow.margin` | the same one |
| `π*` at the exit for missing-outcome fits | `CorrectionRow.observation_margin` | the same one |
| `g_{r,1}` in equation (10)'s covariate, or `γ_a`/`γ_Δ` on a missing-outcome fit | `CorrectionRow.gr1_margin` | **none**. It is a regression of an arm indicator on `Q̄̂`, and `g_0 > δ` does not imply a positive lower bound |

**The two observation rows are not implied by the treatment ones**, which is why they are
separate columns rather than a wider reading of the existing ones. A randomized trial's
treatment mechanism is flat by design and cannot clip, while its observation mechanism is a
fitted probability that can sit at its floor on a large share of rows. A check reading only the
treatment witnesses therefore reports `contract = "theorem"` on a fit that is squarely bound-active.
`tests/unit/test_drtmle_missing.py::TestTheContractSeesTheObservationTruncations` is the pair of
fits that says so: a well-behaved trial reading `"theorem"`, and one whose `π̂` is pinned on a
fifth of its rows reading `"bound-active"` with **only** the two `π` entries named.

The **asymptotic** half of the inactive-bound claim needs three conditions, and only the first is
Theorem 1's.

1. `g_0 ∈ [δ, 1−δ]`. This one is the theorem's.
2. A bound sequence eventually below `δ`. `g_bounds="auto"` supplies it as `5/(√n·log n) → 0`. A
   fixed bound above `ess inf g_0` does not supply it at all.
3. `ĝ` consistent in **sup** norm. This is stronger than the `L_2` conditions the theorem assumes,
   and it is **unverified**.

`CorrectionCheck.contract` reports `"theorem"`, `"bound-active"` or `"none"`. **It is a scope
label and not a verdict**. `CorrectionCheck.passed` deliberately does not read it, because
folding it in would report a perfectly well-solved fit as broken.
