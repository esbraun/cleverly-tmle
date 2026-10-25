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
and therefore did not cover the shipped ten-fold default.

A fixed-seed `glm` check produced the table below. The
counter's historical name is `ill_conditioned`, but it counts every equation-(10) inner failure,
including a tolerance-limited full-rank solve; `res.validate()` therefore describes these as
numerically difficult rather than asserting that every Hessian was singular. The three ten-fold
rows were measured when the default draw balanced the treatment. The shipped draw is unstratified,
so those three rows are dated and the `off` rows are not.

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
one.** The value `"drtmle"` takes the round order of the canonical R package: equation (9), refit only
`g_{r,1}` and `g_{r,2}`, equations (10) and (8), then refit only `Q_r`.

Before the first round, both orders prime `Qbar` with one equation-(8) step. R `drtmle` does not
take that step. Step 2 of the algorithm in Benkeser et al. (2017), Section 3.2, fluctuates the
outcome regression before it fits any reduced regression, and the prime follows it.

`"benkeser"` follows the published
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
draws in each nuisance regime, and the generalisation does not hold. The
[canonical DR-TMLE study](../method-evidence/canonical-dr-tmle.md) was regenerated under the
package's drawn unstratified ten-fold split, and the table below is that measurement. The
[fold and outcome-scale rules](../cv-tmle.md#fold-and-outcome-scale-rules) give the split.

| regime | route difference in `ate`, as a share of one sampling standard deviation | worst draw |
| --- | ---: | ---: |
| both nuisances correct | 7.4% | 34% |
| outcome regression correct | 5.8% | 29% |
| treatment mechanism correct | 45.0% | 196% |

Both routes solve all three equations to about `1e-11` on those draws, so this is not one route
failing to converge. **The three equations do not pin down a single answer when the outcome
regression is misspecified.** Step 7 states its termination as the three empirical means being
approximately zero, and on this law more than one state satisfies that: the route decides which
one is reached. For scale, the disagreement between this package and R `drtmle` in the same cell
is 51.7% of a sampling standard deviation, so *cleverly against cleverly* and *cleverly against
R* are the same size of difference and have the same cause.

The reported `σ²_n` differs for the reason it always did, and that part stands: the exit condition
constrains the three empirical *means*, while the reported variance is the second moment of a
curve built from the reductions, which the two routes leave at different vintages by construction.
The median `|Δσ|/σ` is 0.02% and 0.25% where the outcome regression is right, and 3.02% where it
is not. Both routes ran at the registered `max_outer = 100`.

Read `update_order` as a diagnostic, and read a large route difference as a statement about the
law rather than about the code. The
[canonical DR-TMLE study](../method-evidence/canonical-dr-tmle.md) carries the
measurement.

RM19 measured the prime against R on 2,000 fresh draws of the binary `treatment_correct` law at
`n = 3000`. A transcription of the R loop took the prime and three other named choices each from R or
from this package. Averaged over the settings of the other three factors, the prime moved the mean `ate` by
`-0.000534`. Its interval at the level 0.998 runs from `-0.001298` to `0.000231`. The whole
difference between this package and R reads `no increment at the declared resolution`.
[What the localization design found](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#what-the-localization-design-found) gives each
reading.

## How many rounds

`max_outer` (default 50) caps the alternation. It is **not** `max_iter`, which caps the Newton
steps inside one fluctuation, and the two were confused in this project's own published record:
the registered DR-TMLE manifest listed `max_iter: 100` as its alternation setting while the loop
ran at a hard-coded 50 that no caller could reach. The value that applied is now on
`result.repeats[0].fluctuations["mean"].reduction.max_outer`.

Raising the cap changes only the fits that reached it. Measured on the paper law at `n = 3000`
over 120 draws per nuisance regime, under the package's drawn unstratified ten-fold split, going
from 50 to 100 left all 336 tolerance exits bit for bit and moved the 24 fits that had hit the cap
by up to `1.7e-3`, which is 8% of one sampling standard deviation. Act on a
`"cap"` exit rather than noting it: it says the draw had not settled, and the estimate reported
is the one the loop stopped at. Read it beside `score_check`, which says whether the scores the
fit left are small enough to matter.

## The exits under a declared stopping rule

`Targeting(stopping="scaled_var")` and `"sample_size"` change how the loop judges the outcome
equation. The alternation tests the outcome equation by the rule ratio, the largest score over its
threshold, and it keeps `tol` for equations (9) and (10).
[Stopping rules](../point-treatment-tmle.md#stopping-rules-step-bound-and-best-iterate) defines the
ratio. The default `stopping="relative"` keeps each exit as it was.

A ratio is valid only at the pair of fit and score where it was computed. The `"benkeser"` order
moves the regression and the mechanism after the outcome solve, and so does each round of the
missing-outcome reduction. So these paths restate the score at the pair they judge.

| site | outcome test under a declared rule |
| --- | --- |
| the outer exit and the stall rule, both orders | the ratio restated at the pair after the last move. The outcome entry of the stall score is `tol` times that ratio, so every round uses one ruler |
| the closing judgements, `unsolved` and `converged` | the ratio of the record each one judges, computed at that record's pair. Equation (10) is judged by the ratio of its own columns, because the closing joint solve stops by the rule over them. Equation (9) keeps `tol` |
| the missing-outcome reduction | the ratio restated from the stacked standard and extra scores. The outcome record holds that stacked ratio, and the reduction record holds the extra block's part of it. The drift blocks keep `tol` |
| `"drtmle"` order | the restatement is a no-op, so the restated ratio equals the solver's |
| the closing pass at frozen reductions | no outer outcome test. The outcome solves apply the rule themselves |

`tests/unit/test_stopping_rule_engines.py` recomputes the ratio by hand at every round and compares
it with the ratio the exit read. Under `update_order="benkeser"`, seed 5 and a step of 0.01, the
restated ratio reaches 1.75 while the solver's own ratio is 0.966. An exit that read the solver's
ratio would judge the rule met there. A mutation control that reads it fails the audit.
`score_check()` under a declared rule judges the `(reduced)` row by the reduction record's rule
ratio, and it reports the `D*_Q` and `D*_Y` correction rows as diagnostics. The mechanism rows keep
the default bar. `TestRuleGovernedRows` in `tests/unit/test_stopping_rule_report.py` checks a
one-step fit in each order and a missing-outcome fit, with a mutation control that restores the
default bar. `score_check()` reports no `rule_bound_se` for a DR-TMLE fit, because the corrected
curve carries terms that the outcome columns do not.

The result summary also carries the pinned line for `treatment_adjustment=`, described in
[treatment-mechanism covariates](../point-treatment-tmle.md#treatment-mechanism-covariates).

## The canonical `cvFolds` mapping

R `drtmle`'s `cvFolds` path maps to `cross_fit=True, reduced_crossfit="pooled",
targeting_scheme="pooled", cv_evaluation=False, stratify_folds="none"`. `cvFolds` draws an
unstratified V-fold split, and `"none"` is the only policy a cross-fitted `cleverly` fit accepts. The
[evidence manifest](../evidence.md#estimator-variants-over-registered-targets) states the map, the
test that pins it, and which settings the source audit refuses. The default
`reduced_crossfit="foldwise"` is not the R computation. The R comparator studies pin `"pooled"`.

## Reduced-regression cross-fitting

`reduced_crossfit` sets where fold `k`'s reduced regressions (`Q_r`, `g_{r,1}`, `g_{r,2}`) train.
On a cross-fitted fit, fold `k`'s empirical-process term has conditional mean zero only when those
regressions are functions of the fold-`k` complement. The table defines the two settings.

| setting | fold `k`'s reduced regressions train on | fold `k`'s data reach them |
| --- | --- | --- |
| `"foldwise"` (default) | slab `k`: the production arrays at fold-`k` rows, and leave-two-out arrays `Q̄^(−k,−m)`, `g^(−k,−m)` at the rows of each fold `m ≠ k`. Each tilt moves slab `k` along the covariate built from slab `k` and fold `k`'s reduced regressions | no |
| `"pooled"` | the production arrays `Q̄^(−m)`, `g^(−m)` at the rows of each fold `m ≠ k`. R `drtmle`'s `cvFolds` construction | yes, through the models of the other folds and through the tilts of their rows |

Each fluctuation solves one pooled `epsilon` on the production rows under both settings.
At fold `k`'s own rows, slab `k` holds the production state at every refit, so the estimate and the
influence curve keep their form. `TestEachSlabHoldsTheProductionStateAtItsOwnFold` in
`tests/unit/test_drtmle_crossfit.py` checks this to 0 at every refit.

### The fold leak of the pooled construction

The pooled construction lets fold `k`'s data reach its own reduced regressions through two
channels. The first channel is the initial designs and targets: the models of fold `m` trained on
fold `k`. The second channel is the tilts. A pooled fit moves the row of fold `m` along that row's
covariate, which fold `m`'s reduced regressions built after they trained on fold `k`.

Probe P1 (X83) measured the leak on the `canonical-drtmle` law, with its draws, folds and
learners. The table gives the ATE bias in units of the Monte Carlo SD.

| scenario, n, R | pooled | foldwise | in sample | pooled − foldwise, paired |
| --- | --- | --- | --- | --- |
| `treatment_correct`, 2,000, 400 | +0.339 | +0.134 | +0.049 | +0.227 ± 0.025 |
| `outcome_correct`, 2,000, 400 | +0.116 | +0.027 | −0.012 | +0.091 ± 0.010 |
| `treatment_correct`, 8,000, 200 | +0.108 | −0.018 | −0.034 | +0.130 ± 0.030 |

At n = 2,000 the pooled ATE interval covers 0.915 of `treatment_correct` draws, and the foldwise
interval covers 0.950. The pooled-minus-in-sample bias falls from +0.326 SD to +0.156 SD between
n = 2,000 and 8,000. Standardized bias falls like `n^(−1/2)` when the bias falls like `1/n`, so
the leak is second order. It is large at the sample sizes of the registered studies. The probe
output is not a committed artifact.

The registered
[`drtmle-foldwise` study](../method-evidence/dr-tmle-foldwise.md) fits each draw of its leak pair
twice, pooled and foldwise, on the `canonical-drtmle` law. Under `treatment_correct` at n = 1,000 and
4,400 replications, its ATE bias is +0.400 SD pooled and +0.156 SD foldwise. The paired
displacement is 0.266 SD, against a declared floor of 0.10.

### The argument for the foldwise construction

Write `ĥ_k` for the contribution of fold `k`'s reduced regressions, and `e` for the history of the
pooled `epsilon` values. Given `e`, slab `k` and fold `k`'s reduced regressions are functions of
the fold-`k` complement. So the term that the pooled construction leaves open is zero. The
remaining term needs the three conditions in the table.

| condition | content |
| --- | --- |
| (E-eps) | given the complement, the class `{h_k(·; e)}` indexed by `e` has bounded entropy at `P_0` |
| (C) | `ê = O_p(1)` |
| (L2) | `h_k(·; ê)` is `L_2` consistent at the targeted state. Theorem 1 assumes it |

With these conditions, the conditional Donsker step (van der Vaart 1998, Lemma 19.24) gives
`(P_{n,k} − P_0) ĥ_k = o_p(n^(−1/2))`. The dimension of `e` is at most `max_outer` times the
number of equations, arms and stratum blocks. The random round count makes the class a finite
union.

(E-eps) follows from condition (E) below by composition with the smooth tilt family. It needs
entropy at `P_0` only, because the design map is fixed given the complement.

> **(E)** the reduction learner's fitted functions of one or two scalars lie, with probability
> tending to one, in a class whose bracketing entropy is bounded uniformly in the underlying
> measure.

The reductions have fixed dimension: one scalar for the default construction, and two for the
bivariate reduced probability. A fixed-basis linear or logistic model is a bounded sieve of that
dimension. Histogram boosting with fixed iteration and leaf limits is a bounded-variation ball. A
CV-selected round count takes boosting out of the class. A random forest is outside the class.

Zheng and van der Laan (2010) give the pooled-`epsilon` structure. Their fold fit is a parametric
submodel, and here the covariates are learner refits. No cited result proves (E-eps), so it is an
assumption of this construction.

The foldwise construction keeps two second-order terms.

| term | why it is second order |
| --- | --- |
| fold `k`'s reduced regressions predict in sample at slab `k`'s training rows, to build the slab covariate there | in each one-wrong regime the scores of equations (9) and (10) have mean zero at the consistent nuisance, so their `epsilon` goes to 0. Equation (8)'s covariate `1/g` reads leave-two-out primaries only |
| the leave-two-out fits train on `(K − 2)/K` of the rows, and the production fits on `(K − 1)/K` | the difference enters as the difference of two cross-fitted fits of one nuisance times the error of the consistent nuisance |

### Tests of the construction

| test (`tests/unit/test_drtmle_crossfit.py`) | what it pins |
| --- | --- |
| `TestAFoldsReducedRegressionsNeverSeeThatFoldsData` | with the `epsilon` history fixed, flipping fold `k`'s outcomes and permuting its treatments leaves slab `k` unchanged at every row and refit, in both update orders. Restoring the production covariate of equation (8), (10) or (9) for the slabs breaks it |
| `TestTheSlabsTiltAlongTheirOwnCovariates` | with the slabs carried along the production covariates, a foldwise fit equals a pooled fit bit for bit |
| `TestADataIndependentPrimaryLearnerLeavesOnlyTheTiltChannel` | with identical initial designs, the two settings still give different estimates |
| `TestEachSlabHoldsTheProductionStateAtItsOwnFold` | own-fold slab rows equal the production state at every refit, with strata and with an `evaluation=` companion |
| `TestThreeArms` | the same identity at three arms in both update orders. Each pair of folds is fitted once per learner, and the longhand fit outside the pair reproduces both of its slab entries |
| `TestAScreenedMechanismIsScreenedInEverySlab` | with `screen_treatment=True` each leave-two-out mechanism equals a screened fit outside its pair. An unscreened fit differs by more than `1e-3` |

### Limits of the foldwise construction

| limit | statement |
| --- | --- |
| folds | `n_folds` must be at least 3. A fit with fewer folds raises `CapabilityError` (`INVALID_DECLARATION`) before any learner, and names `reduced_crossfit="pooled"`. The fit asks again when `n_folds` changes after construction |
| trainable rows | each pair of folds must leave a trainable row of every arm in the whole sample. The leave-two-out primary regressions do not train inside a baseline stratum. A split that does not leave such a row is refused after the fold draw and before any learner |
| `targeting="one_step"` | refused on cost: the one-step walk would carry one slab per fold. The refusal names `targeting="iterative"` and `reduced_crossfit="pooled"` |
| truncation | a slab takes the pooled `epsilon` without a solve of its own. Without a real truncation a slab's mechanism can run to 0. A cross-fitted fit at `g_bounds=(1e-9, 1 − 1e-9)` on the informative cluster law reaches the outer cap with `g_{r,2}` near 3e7 and a standard error near 1.4e5. The fit reports `failure="max_iter_reached"` and raises a `ConvergenceWarning`. The pooled fit of the same draw exits on tolerance (`TestAnAlternationThatRunsOutOfRoundsIsAFailure` in `tests/unit/test_drtmle_crossfit.py`) |
| saved results | a foldwise result carries `K` slabs of primary arrays and reduced sets, so a saved artifact grows about `K`-fold. A retarget, such as `truncation_curve()`, reads the slabs |
| `targeting_scheme="fold"` | refused. The foldwise construction is pooled in the tilt |
| in sample | `cross_fit=False` has one fold and no complement, so both settings run one in-sample path |

### Cost of the foldwise construction

The foldwise construction fits each primary nuisance once more for every unordered pair of
folds. One fit serves both slabs of the pair. With `n_folds=K` that is `K(K − 1)/2` extra fits of
each primary learner. `TestThreeArms` in `tests/unit/test_drtmle_crossfit.py` counts the fits.
Pass explicit learners and fewer folds when the default learner library is too slow.

The table gives one measurement. The setting is the `both_correct` law of
`tests/studies/canonical_drtmle.py` at n = 500, the default learner library, the default 10 folds,
and one core. The measurement ran before the pair cache. That implementation made `K²` extra
fits of each learner: it fitted each pair twice and refitted each production model. At `K = 10`
the extra fits now number 45 per learner in place of 100. The time was not measured again.

| phase | `"pooled"` | `"foldwise"` |
| --- | --- | --- |
| initial nuisance fits | about 80 s | about 910 s (100 leave-two-out fits per learner) |
| first refit of the reduced regressions | 192 s | 195 s |
| one outer round after the first | not measured | about 200 s |

The foldwise fit did not exit on tolerance within 30 outer rounds (cap `max_outer=50`). Each round
refits the default reduced learners on every slab. The pooled fit had not finished after 2 hours
on a shared machine either, so the round count is a property of the default reduced learners, not
of the construction. The `drtmle-foldwise` study passes explicit reduced learners for this reason.

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
