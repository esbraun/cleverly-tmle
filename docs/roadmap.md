# Roadmap

What has landed, what is open, and why native acceleration is not worth building.

**One thing is open**: `DRTMLE`, the doubly-robust-inference variant, which is written and
tested and not finished. [What is still open](#what-is-still-open) is the list, grouped into
four pieces of work, each of which is a pull request rather than an errand. That grouping and
its order are a revision: an [external review](drtmle-review.md) of this page and the code
behind it read the plan against Theorem 1 of Benkeser et al. (2017) and found the definition
of done right and the route to it short by two conditions, which are now items 13 and 15.
A fifth piece, **0**, has landed and is in [closed since this list
opened](#closed-since-this-list-opened).
Everything else on this page is a record: [Refusals worth
lifting](#refusals-worth-lifting) is the list of parameters this package had the machinery for
and had simply not written down, and it is now empty; [What the sizings got
wrong](#what-the-sizings-got-wrong) is what estimating that work taught, kept because the next
sizing is the only thing it is for.

Nothing is queued behind `DRTMLE`. The remaining rows under [Not written
yet](methodology.md#not-written-yet) are there because nobody has asked, not because anything
stands in the way, and the one standing conditional item is the [HAL
trigger](#on-native-acceleration).

## Variants

Estimators that plug into the shared base classes (`estimators/base.py`, `inference/`,
`learners/`, `fluctuation/`), as against further estimands, which plug into the target
registry.

- **longitudinal TMLE (`cleverly.longitudinal.LTMLE`) — landed.** Static regimens and
  **dynamic rules** `d_t(H_t)`, time-varying confounding and monotone censoring, a **survival
  outcome** reporting the cumulative risk curve with joint bands over it, **competing risks**
  reporting a cause-specific cumulative incidence per cause, observation weights, and a
  **working model over the regimens**. See [treatment given over
  time](user-guide.md#treatment-given-over-time), [a survival
  outcome](user-guide.md#a-survival-outcome), [competing risks](user-guide.md#competing-risks)
  and [summarising the
  regimens](user-guide.md#summarising-the-regimens-a-marginal-structural-model); what it
  refuses is listed there under a `kind` column. The largest of those refusals is the *other*
  competing-risks estimand — the incidence under **elimination** of the competing events,
  which intervenes on them rather than conditioning on the history, and so is [a different
  question](methodology.md#a-different-question) rather than a gap: a further factor per node
  in the denominator, and its own identification
- **doubly-robust nonparametric inference (`cleverly.DRTMLE`) — in progress.** van der Laan
  (2014); Benkeser, Carone, van der Laan & Gilbert (2017); Benkeser & Hejazi (2023). Every
  interval reported here is valid when the second-order remainder is negligible, which needs
  *both* nuisances converging fast enough; `DRTMLE` buys an interval that stays valid when only
  one of them is consistent, by estimating reduced-dimension regressions of each nuisance's
  residual on the other and solving their score equations too. The derivation, the equations
  and what each guard removes are in [what the extra equations
  remove](methodology.md#doubly-robust-inference-what-the-extra-equations-remove); how to call
  it is in [doubly-robust inference](user-guide.md#doubly-robust-inference).

  The code is written, in five commits, and every test passes. That is not the same as
  finished, and calling it landed would be claiming the part that is missing: **Theorem 1 of
  Benkeser et al. (2017) is unread**, so the influence curve — which is the whole of what this
  variant is for — is transcribed from R's `drtmle` rather than derived; nothing here has been
  compared against that package's numbers; a coverage study found **no gap for the variant to
  close** at the sizes it could reach; **under weak overlap the score check fails on 23 of
  24 fits**, so do not use this estimator where overlap is poor; and — the one to read if you
  read only one — **on roughly a quarter of splits the reported curve is not centred**, at
  `2e-05` to `7e-04` where a solved fit sits near `1e-09`, while the loop's own three rows all
  report their scores solved to `1e-11`. That is item 20, it is *not* confined to poor overlap,
  and until it is understood a `DRTMLE` standard error should be read as provisional on every
  process. It is caught, on the influence-curve rows: `res.score_verdict` says so and
  `summary()` prints it, which is the only reason it was found at all. [What is still
  open](#what-is-still-open) is the rest, and three of its items are there because an
  [external review](drtmle-review.md) put them there: the theorem's *other* assumption beyond
  the three score equations (item 13), the cross-fitting construction the reductions would
  need to satisfy it (item 15), and the claim that the alternation converges (19). Four more
  of the review's items — that the corrected curve is the efficient one (14), that a fit whose
  score check fails still reports an ordinary Wald interval (16), that weights need nothing
  said about them (17), and that `repeats=` averages what it is averaging (18) — were piece 0
  and have [landed](#closed-since-this-list-opened). A fit now says on its own report when its
  score check fails, which is how item 20 was seen at all

## What is still open

**Done still means one thing: a demonstration that `DRTMLE`'s interval attains its nominal
coverage where a plain `TMLE`'s does not.** That is [piece C](#c-the-demonstration) below, and
nothing less clears the variant. What the [review](drtmle-review.md) changed is not that bar
but what it takes to *believe* a demonstration that meets it. Coverage is one link of five, and
each link can hold while another fails:

1. **Theorem fidelity** — the equations solved and the curve reported are the ones the
   derivation gives, under conditions the fit actually meets. Items 1, 13 and 15.
2. **Reference fidelity** — the algorithm agrees with `drtmle` component by component, not
   merely at `psi` and `se`, where several differences cancel. Item 2.
3. **Independent correctness** — a check that could not agree with `drtmle` by copying it.
   Half of it exists: `tests/unit/test_remainder_drtmle.py` is the exact-law arithmetic. The
   other half is item 13's empirical rate.
4. **Numerical validity** — every required score solved to a statistically negligible order,
   *at the arrays the reported curve is built from*, and a fit that fails to say so somewhere a
   reader cannot miss. Items 11, 12, 16 and 20. The qualification is item 20's: a fit can solve
   all three equations to `1e-11` by its own record and still report a curve whose mean is
   `2e-04`, which is a link-4 failure that announces itself nowhere in the loop's own
   diagnostics. Item 11 is the loud version of this link failing and item 20 the quiet one, and
   the quiet one is on a quarter of ordinary splits rather than on a regime a user was told to
   avoid.
5. **Inferential usefulness** — coverage in a regime where the plain interval fails. Item 3.

The review's own summary of this is exactly right and worth keeping in its words: none of the
five implies the others. A `psi` that matches R proves nothing about the variance, which is the
only thing this variant produces; a curve that matches the theorem proves nothing about whether
the alternation solved its equations on a given draw; and the first four together prove nothing
about whether the interval is ever *better* than the one `TMLE` already reports. The
[limitations](#limitations-recorded-rather-than-fixed) after the pieces are outside the chain
entirely: real, understood, and unable to move a coverage number. Anything that *can* move one
belongs in a piece, which is where item 20 went after being filed there by mistake — see
[lesson 7](#what-the-sizings-got-wrong).

That bar is deliberately higher than "no known defects". An estimator with every limitation
below resolved and no demonstration is one that computes something nobody has shown is worth
computing. The variant exists for exactly one claim — `TMLE` is doubly robust for *consistency*
and singly robust for *inference*, and this closes the second gap — and a package that ships it
without evidence is asking to be believed rather than showing its work. `bench_tmle.py`'s
conclusion that a Rust extension was not worth building counts as a result because it was
*measured*, and the same standard cuts both ways here: a study that finds no gap at reachable
`n` is also a clearing outcome, and the honest response is to say so in the README rather than
to keep looking.

The numbered items the pieces below close keep the numbering they have had since the list
opened, because `benchmarks/bench_drtmle.py`, `.github/workflows/drtmle-convergence.yml`,
`estimators/targeting.py` and `tests/unit/test_drtmle_fit.py` all cite them by number. **The
review's items are therefore 13 to 19 rather than a renumbering**, even where one of them is
more important than an item with a lower number, and item **20** — found while closing item 18
— is 20 for the same reason. The pieces are lettered so the two cannot be confused.

### The work, in four pieces

Each is a pull request. Small items are grouped where the *evidence* is shared — piece B is
five items because one dispatch of the same sweep answers all of them — not where the subject
matter merely rhymes.

#### A. Check the curve against something other than itself

**Closes items 1, 2 and 15, and opens item 13.** The influence curve `D = D* − D*_Q − D*_g` is
read off `drtmle`'s implementation, not derived. The whole variant is a variance estimate, so a
curve transcribed from software and never checked against its derivation is the one part of this
that could be wrong in a way nothing here would catch.
`inference/influence.py::reduced_corrections` says so in its own docstring, as do the guide and
the appendix.

Two halves. They were one piece of work because they answer the same question; **the review
splits them, and it is right to**, because only one of them is blocked. A1 needs a paper this
environment cannot fetch. A2 needs neither the paper nor R's agreement to be worth doing, and
it is what should be written first.

**A1 — the theoretical audit.** *Closes items 1 and 15.*

- **Read Theorem 1 of Benkeser et al. (2017).** The only thing on this page that has to happen
  outside the repository: Biometrika is paywalled and this environment's network policy denies
  the working-paper mirrors. If the theorem and the transcription disagree, the theorem wins
  and `reduced_corrections` is wrong.
- **Write the concordance.** One table, permanent, mapping each object of the theorem to its
  Python name and its R name, and stating for each: the conditioning variable, the sign, the
  denominator, whether the value is initial or starred, and whether it is arm-specific. The
  review drafts it and the draft is usable as-is. It is the artefact that makes the next
  reader's audit cheap rather than a re-derivation.
- **List the assumptions and say which the implementation meets**, rather than only checking
  the formulas — positivity, the nuisance limits, the reduced regressions' rates, **the
  appendix's remaining second-order terms**, **the empirical-process condition**, whether the
  equations must be solved exactly or only to `o_p(n^(−1/2))`, whether the theorem covers the
  arm-level means, the ATE contrast, or both, and whether it covers weights (item 17) and `K`
  arms ([piece D](#d-widen-the-scope-to-what-the-sources-derive)). The two in bold are already
  known to be open questions rather than boxes to tick, which is why they are numbered:

**13. The theorem asks for one thing more than the three score equations, and it is unmeasured
here.** Solving equations (8), (9) and (10) is *necessary*; Theorem 1 separately assumes the
remaining second-order terms are `o_p(n^(−1/2))`. Nothing on this list checked that, and
coverage in piece C could come out right without it — which would be an accident nobody could
distinguish from a result. Half the check exists already:
`tests/unit/test_remainder_drtmle.py` shows on an exact law that one guard removes the whole
first-order part of `R₂`, that the unguarded remainder is not already zero, and that coarsening
a reduction leaves a residue — the arithmetic, at saturated reductions. The missing half is
empirical, at *estimated* reductions:

```text
R_remaining = psi-hat − psi_0 − (P_n − P_0) D-hat_DR
```

computed at a known truth, and shown to satisfy `√n · R_remaining → 0` across sizes in **both**
off-diagonal regimes. That is a column on piece C's study rather than a run of its own, since
that study is the only place that knows `psi_0` and already fits both estimators at three
sizes — so item 13 is opened here, where the reason for it is, and closed there.

**15. The reduced regressions' cross-fitting is defended in an implementation note, not an
argument.** `fit_reduced`'s docstring is the most careful writing in this variant and it
reaches the right conclusion for the wrong kind of reason: it reuses the primary split, shows
that an independent split removes *none* of the induced dependence (the contamination is in the
design values, not in which rows train), and shows that per-fold designs would trade a
second-order dependence for a first-order covariate shift. All of that is sound and none of it
establishes what the theorem needs, which is that the induced dependence is higher order in the
expansion. The review's framing is the one to adopt: *determine a construction satisfying the
empirical-process conditions of the DRTMLE expansion, and say whether fold reuse is one.* The
candidates are nested cross-fitting, three-way splitting, per-outer-fold reductions, and the
current pooled construction with a proof. Agreement with R settles nothing here — that package
predates the construction — so this is A1's work and not A2's. If the answer is "pooled is
fine, and here is why", the docstring gains a paragraph and the item closes; if it is not, the
expensive nested version is the reference implementation to measure the cheap one against.

**A2 — reference and independent validation.** *Closes item 2.* Where the review is most
directly useful: the current plan's "one fixture fit, its `psi` and `se` committed, one test"
is too coarse to catch what it is for. Several differences cancel at `psi` — a sign error in
one correction, a scaling, a swapped `gr1`/`gr2`, targeting at the wrong starred arrays — and
`psi` is precisely the quantity all three empirical means being zero makes *insensitive* to the
corrections. Compare **components**: the initial `Q̄` and `g` predictions, each reduced
regression, each targeting coefficient, `D*`, `D*_Q` and `D*_g` separately, the full corrected
curve, the three empirical scores, then `psi` and `se`. Start from user-supplied nuisance
arrays or a deterministic GLM, not a Super Learner, so a discrepancy is arithmetic rather than a
fold draw. And include a **deliberately misspecified** fixture, because at the truth `Q_r` and
`g_{r,2}` vanish row by row and a broken implementation agrees with plain `TMLE` and with R
alike — the degeneracy [lesson 2](#what-the-sizings-got-wrong) is about, arriving here for the
third time.

The second half of A2 is the independent check, and the reason it is not optional is that a
cross-language comparison can reproduce a shared bug: both packages descend from the same
source. `tests/unit/test_remainder_drtmle.py` is already that check for the *guards*; what it
does not do is pin the reported curve's own decomposition against a perturbation of the law the
way the `test_influence_gateaux*` modules do elsewhere — and those modules cannot be reused
here, derivably, because everything the variant adds vanishes on an exact law.

**One trap for anyone reading the R source alongside the paper**, and it is why this piece is
not the errand it looks like: `grn1` there is the paper's `gr2` and `grn2` is the paper's
`gr1`. The numerator and denominator roles are swapped between the two, so a formula
transcribed from one and checked against the other is inverted and still plausible.

When **A1** lands, the labels change with it: `reduced_corrections`, the [methodology
section](methodology.md#doubly-robust-inference-what-the-extra-equations-remove) and the guide
all currently say **what `drtmle` computes** rather than what the theorem derives, and that
wording is load-bearing until it closes. A2 landing on its own changes none of it — parity with
that package is evidence about the transcription, not about the derivation, and the review is
right to insist the three claims stay separate: that Python implements the same algorithm, that
the algorithm satisfies the theorem, and that it helps in finite samples.

#### B. The loop's exit, and whether what it leaves is what gets reported

**Closes items 11, 12, 19 and 20, and re-measures items 4 and 6.** Six things, and they are
one piece because one dispatch of `benchmarks/bench_drtmle.py` produces the evidence for all of
them.

This piece used to be called *weak overlap, and the loop's exit under the rule it now uses*,
and item 20 is why it is not. Weak overlap is one regime where the loop leaves the equations
unsolved and says so loudly; item 20 is a quarter of *ordinary* splits where it leaves them
solved at arrays the reported curve is not built from, and says nothing at all. The two are the
same question — is what the loop leaves what gets reported — and only one of them is about
overlap.

*The fit does not solve its own score equation under weak overlap, and nothing said so before
the sweep.* On `weak_overlap_dgp` the score check fails on **23 of 24** fits, with the worst
score at rough parity with `se/√n` — median `1.1` at `n = 600` and `1.0` at `n = 1,200`,
against `1e-7` to `4e-7` on every other process. A score the size of `se/√n` is not a tolerance
that wants loosening: it is first order in the very quantity the interval is built from, so the
interval is not the one the derivation describes. It is **not** the exit criterion — refit under
both rules the same two draws fail with worst scores agreeing to three figures (`1.65e-3`
against `1.64e-3`, and `7.97e-3` against `7.96e-3`) — and it is **not** the conditioning of item
4, since `ill_conditioned` is 0 of 24 here, the only process where it never fires. What it most
likely is, and what has *not* been checked, is that `1/g` under weak overlap makes equation
(8)'s covariate so large that the truncation is doing the work; `sensitivity.positivity()` on
these fits is the obvious next reading. Until then this is the item that should stop anyone
using `DRTMLE` where overlap is poor.

**The diagnosis has been too narrow, and the review widens it usefully.** `1/g` in equation (8)
is one of *five* places weak overlap enters, and the sweep should read all of them before
concluding: equation (9)'s covariate is `Q_r/g`; `g_{r,2}`'s own *target* is a quotient by `g`,
formed once at fit time (item 9); the ratio `g_{r,2}/g_{r,1}` is unstable when either the
numerator is noisy or the denominator small; and truncating `g` moves not just the covariates
but the reduced regressions' estimands, since two of the three condition on `ĝ`. So the failure
may be a structural incompatibility between aggressive truncation and the equations rather than
a solver that stops early — and those two have different remedies. What the sweep should record
per fit, beyond what it records now: quantiles of raw and truncated `g`, per-arm effective `n`,
the high quantiles of every clever covariate, the distributions of `Q_r`, `g_{r,1}`, `g_{r,2}`
and their ratio, the share of each score carried by the top 1%, 5% and 10% of rows, the Hessian
condition numbers, the scores either side of truncation, `psi` and `se` across a truncation
grid, and whether the failures persist when the reductions are handed the *oracle* values. That
last is the one that separates a noisy reduction from a wrong equation, and it costs nothing
because the datasets already know their truth.

**The product decision belongs to this piece.** If the sweep finds no stable region, `DRTMLE`
should refuse or invalidate under weak overlap on a **predeclared** diagnostic rather than warn
— a warning is easy to miss, and this is a method whose only purpose is inference. The
reporting half of that is [item 16](#closed-since-this-list-opened), which has landed; what
this piece adds is the threshold and the name of the state, decided from evidence.

*The sweep measures the criterion that was replaced.* [The table](#how-the-alternation-exits) is
the evidence item 7's change was argued from, which is the right way round — the failure had to
be characterised before the threshold moved — but it means the exit distribution under the
current rule is uncharacterised. A rerun is one dispatch and about 45 minutes, and would say
whether `tolerance` is now the norm at scale or only on the six fits looked at. It re-measures
items 4 and 6 for free.

*The absolute bar is a proxy for the one it cites.* `score_check` compares against
`DEFAULT_TOLERANCE * se / sqrt(n)` using the fit's actual `se`; `targeting._solved` substitutes
`_NEGLIGIBLE / n`, which assumes `se = O(n^-1/2)` on the scaled outcome rather than measuring
it. It is conservative exactly where it matters — under weak overlap `se` is large, so the
loop's bar is the stricter one — but "conservative on the cases we looked at" is not "correct",
and a fit with a very small `se` is the untested direction. Passing the realised `se` in would
remove the assumption; it was not done because the loop runs before the estimate exists.

The review's way out of that circularity is better than passing `se` in, and it is the design to
build: **the loop's bar should not be a proxy for the reported one at all.** Asymptotic
linearity asks for `P_n D = o_p(n^(−1/2))`, and the honest finite-sample rendering of `o` is a
deterministic sequence `c_n/√n` with `c_n → 0` slowly — a *numerical* criterion, stated as one,
with the standardised score `|P_n S_j| / sd-hat(S_j)` reported afterwards as a separate
diagnostic rather than folded into the stopping rule. That separates the two things `_NEGLIGIBLE
/ n` currently conflates: when to stop iterating, and whether the fit that came out is entitled
to a Wald interval. The second is `score_check`'s job and [item 16](#closed-since-this-list-opened)'s.

**19. The alternation's convergence argument proves less than it is read as proving.**
`solve_with_reduction`'s docstring argues that equation (9) is a weighted logistic MLE of
`A | W` and equations (8) and (10) are the outcome quasi-likelihood — separate factors of the
likelihood of `(A, Y) | W` — so each step maximises its own factor with the others held fixed
and "the joint value never decreases". The review reads the mid-loop refit of the reductions as
breaking that, and it does not: the reductions enter as the *directions* of the submodels, not
as values of the objective, so refitting them changes the next step's direction and leaves the
current joint value where it is, and monotonicity survives. What does not survive is what the
argument is used for. A bounded monotone sequence converges *in value*; that is why the loop
terminates, and it is not why the iterates approach a common zero of three score equations —
under a direction that changes each round, the fixed point of the ascent need not be a
stationary point of anything. The sweep already shows the gap in numbers: **86 of 96 fits
stalled** at a point the objective would not climb from, against 2 that reached the tolerance.
So the wording is the fix — state it as an estimating-equation iteration with empirical
convergence diagnostics, keep the monotonicity claim for what it does buy (termination, and the
reason not to restart from `Q̄⁰`), and drop the implication that stalling is a numerical
disappointment rather than the expected exit.

**20. On a quarter of splits the reported curve is not centred, and the fluctuation rows say
it is.** Found by checking item 18 and not by looking for it. Over 24 draws — twelve
`repeats=2` fits on `nonlinear_dgp` at `n=600` with `glm` on both nuisances, `n_folds=5`,
`learner_folds=3` — **six** leave `Pn[D*_Q + D*_g]` above `1e-8`, at magnitudes from `2e-05` to
`7e-04` on the scaled outcome, every one of them exiting on `"tolerance"` with no failure
recorded and no ill-conditioned round. On one such draw equation (9)'s **recorded** score is
`3.7e-11` while the mean of the `D*_g` the curve actually subtracts is `-2.3e-04`, so the two
are measured at arrays that are not the same. That is the class of defect
`TestTheCurveReadsWhatTheAlternationLeft` exists for, and nothing here covered it, because
every other test in the module reads one fit on one split.

This is not a `repeats=` defect and refusing that keyword would misdiagnose it: a draw of a
repeated fit is an ordinary fit, and the affected draws include first draws.
`tests/unit/test_drtmle_fit.py::TestTheReportedCurveIsNotAlwaysCentred` pins the numbers.

It closes here because which arrays the closing pass leaves the three equations solved at is
what this piece is about, but it is **found** in
[A2](#a-check-the-curve-against-something-other-than-itself), and that is the whole reason A2
goes first: a divergence between two arrays that should be equal is what a
component-by-component comparison locates and what no amount of re-sweeping will. The sweep
this piece dispatches can *count* it — add `Pn[D*_Q + D*_g]` at the reported arrays to the
per-fit record, beside the three recorded scores, so the disagreement is a column rather than
an anecdote from twelve fits on one process — but counting it is not finding it.

`score_check` **does** catch it, on the *influence-curve* rows, which are computed from the
curve rather than from what the solver recorded — so a fit in this state now says so on its own
report rather than printing an interval like any other. That is item 16 arriving on the first
case nobody constructed for it, and it is the only reason this was seen.

**Where this piece sits in the order** has changed. It was first, because poor overlap is a
natural way to make the remainder bite and so may be exactly where piece C has to look, and
because it is the one regime where the score equations are not solved at all. Both remain true.
Against them the review makes the stronger point: item 11's failure is measured *through the
reported curve*, so a curve that is wrong makes the diagnosis uninterpretable — and the same
holds of piece C's coverage. The resolution is not to reorder wholesale but to notice that only
**A1** is blocked outside the repository. [A2](#a-check-the-curve-against-something-other-than-itself)
is cheap, needs nothing this environment cannot reach, and settles the question B's evidence
rests on; it goes first, then B, then C, with A1 the moment the paper can be got. Item 20
sharpens that: a curve whose mean is `2e-04` where its own solver reports `1e-11` is a
divergence between two things that should be the same array, which is exactly what a
component-by-component comparison finds and what no amount of re-sweeping will.

#### C. The demonstration

**Closes item 3, and item 3 is the definition of done.** A coverage pilot over the off-diagonal
of the misspecification grid put `TMLE` and `DRTMLE` at 0.958 apiece in one cell and 1.000 in
the other — no gap to close. The diagnosis is understood: a correctly specified *parametric*
nuisance converges at `n^(−1/2)`, so `R₂` is `O(n^(−1))` and the product condition never binds.
There was nothing for the variant to fix. `tests/e2e/test_coverage_slow.py`'s
`TestDoublyRobustInference` guards what it can — that the point estimate is still doubly robust,
that the interval does not *cost* coverage, that the standard error matches the spread of the
estimates — and says in its own docstring that it is not a demonstration.

The remainder is `R₂ = ‖ĝ − g₀‖ · ‖Q̄̂ − Q̄₀‖` and a `TMLE` interval needs `√n · R₂ → 0`, so the
regime wanted is one where that product does *not* vanish fast enough while one nuisance is
still consistent. Four things the study has to contain, and each rules out a way of being
believed for the wrong reason:

- **Both off-diagonal cells**, not one. `Q̄` right and `g` wrong, and `g` right and `Q̄` wrong.
  `DRTMLE` should hold nominal in both; `TMLE` should fall short in at least one. One cell is an
  anecdote, because which nuisance is wrong is the whole axis.
- **A genuinely slow nuisance** — an adaptive learner whose rate is worse than `n^(−1/4)`. This
  is what the pilot lacked, and it is what makes the study expensive rather than merely long.
- **Coverage against its Monte Carlo standard error**, over replications. `CLAUDE.md`'s rule
  applies with force: never assert coverage on a single fit, and size the replication count to
  the gap being resolved. Separating 0.95 from 0.88 wants a few hundred, not 120.
- **A size trend.** The claim is asymptotic, so the gap should *open* as `n` grows. Two sizes
  showing `TMLE` drifting down while `DRTMLE` holds is far better evidence than one size showing
  a difference, and it is what rules out a coincidence at a single `n`. **Three, if the budget
  reaches** — the review is right that two sizes are suggestive and three carry a rate.

Two additions the review makes, and both change what the study has to *contain* rather than how
big it is:

**"A flexible learner in enough dimensions" is not a rate, and a coverage gap is not a
mechanism.** The second bullet above names the property wanted and no way to know it was
achieved; a Super Learner's realised rate is neither identified nor reproducible, so a gap it
produces could as easily be finite-sample instability as the intended drift. Build the slow
nuisance out of something whose rate is *prescribed* — a series or spline regression with a
chosen smoothing sequence, a histogram at a set bin count, a learner whose approximation bias
is stepped down with `n` on purpose — and keep the Super Learner for the applied stress tests
that come *after* a demonstration, where it belongs. Then **verify the regime was entered**, per
size, against the truth the DGP knows: `‖Q̄̂ − Q̄₀‖` and `‖ĝ − g₀‖` with their log-log slopes, the
misspecified nuisance's distance to *its own* limit staying bounded away from zero, and
`√n · R₂` failing to vanish for `TMLE` while item 13's `√n · R_remaining` does vanish for
`DRTMLE`. Those columns are what turn a coverage table into a demonstration; without them a
correct number is still only a number.

**Report enough to be argued with.** Per estimator, per cell, per size: bias, `√n` bias,
empirical sd, mean estimated `se` and their ratio, coverage, interval width, rejection rate
under a null variant, targeting-failure rate, the share of intervals item 16 marks invalid, and
a Monte Carlo standard error against every one of them.

**Item 20 is a design input here, not a footnote.** On present evidence roughly a
quarter of draws report a curve that is not centred, and that is on `nonlinear_dgp` with good
overlap — not in the weak-overlap cells where this page otherwise leads a reader to expect
invalid fits. So the invalid share this study reports may be ~25% in the cells the
demonstration turns on. A coverage number computed over the surviving three-quarters is
conditional on a non-random subset, selected on a diagnostic that is correlated with the fit
having gone wrong, and reporting it as *the* coverage would be the same class of error as
reporting a per-protocol analysis as intention-to-treat. The rule therefore has to say what
happens to an invalid fit — counted as a miss, excluded with the exclusion rate reported beside
every number, or the study held until item 20 is fixed — and it has to say so before the
numbers exist. The last of those three is the honest default: a demonstration whose exclusion
rule was chosen after seeing which cells it helped is not a demonstration.

**Predeclare the decision rule** before the dispatch: `DRTMLE` compatible with 0.95 at the largest size in both cells, `TMLE` short by a
practically meaningful margin in at least one, an `se` ratio near one, failure rates low, and
the whole thing surviving a second independent seed batch.

One trap in building it, already met once: `tests/e2e/test_double_robustness.py`'s "correct"
cell is an **oracle** (`OracleOutcomeContinuous`, `OracleTreatment`), which makes the good
nuisance exactly right, `R₂` exactly zero and `TMLE`'s interval already valid. The gap opens only
where the good nuisance is *estimated*.

**"No gap found" remains an honest outcome, and the review sharpens what it obliges.** The
existing rule is to say so in the README rather than keep looking. The addition: in that event
`DRTMLE` does not become a production feature by default — it stays experimental or leaves the
public API until some operating regime is demonstrated. A variant that ships with "we looked and
found no case where this helps" in its own docs is not a neutral state.

**What it costs, since that is why it has not been done.** A `DRTMLE` fit is 43s at `n = 1,200`
(measured, [the sweep](#how-the-alternation-exits)) and a study runs both estimators over every
replicate. Two cells by two sizes by 250 replicates is ~2,000 fits, which is ~24 hours serial
and about two on a 12-way `matrix:`. A third size and the nuisance-rate columns roughly double
it. That is a dispatch-only workflow of its own — `drtmle-convergence.yml` is the template — and
the nightly tier must not absorb it.

#### D. Widen the scope to what the sources derive

**Closes the two candidates in item 10.** Everything else in that item is a refusal with a
reason, not a gap. Both candidates are gated on reading rather than on writing, which is why
they sit behind **A1** — the reading — rather than beside it.

- **`reduction="bivariate"`.** van der Laan (2014)'s original single bivariate
  `gr(a | w) = P(A = a | Q̄̂(a, W), ĝ(a|W))` in place of the `gr1`/`gr2` pair, with equation
  (10′) in place of (10). It is derived in the sources and was in scope; it was cut because it
  is a different extra equation on a two-column design rather than a wider loop over the first,
  and nothing was waiting on it. **"Derived" is not "transcription"**, and the review is right
  that calling it that before the score, the correction term and the targeting step are mapped
  would be repeating exactly the mistake items 1 and 2 exist to fix. It gets its own reduced
  object, its own submodel and its own fixtures rather than being folded into `ReducedSet`'s
  array schema — two reductions whose estimating equations differ structurally should not share
  a container — and it is worth a side-by-side run against the univariate pair, since which of
  the two is better behaved on a real fit is not something either source settles.
- **A multi-valued treatment.** The obvious reading of the source says this is already licensed
  — `drtmle(a0 = c(0, 1, 2))` reports treatment-specific means at `K` arms and the software
  paper works an example, the estimating equations are written with a free `a`, and nothing in
  them has a two-arm step. What is missing is the derivation: van der Laan (2014) states its
  problem for a "subsequently assigned **binary** treatment", and no theorem read here covers
  `K` arms. An implementation that accepts an argument is not a proof that the argument is
  licensed, and the gap is not hypothetical — the per-arm mechanism tilts do not renormalise, so
  the targeted `g*(·|W)` at `K` arms is not a distribution over the arms, and whether that is
  harmless is exactly the sort of thing a theorem would say and an example would not. What would
  settle it is reading the multi-arm case in the 2017 paper; if it is there, the widening is a
  wider loop plus a multi-arm mechanism tilt, which `solve_mechanism` does not have, since
  `ipsi` declares `requires_binary_treatment` and has never needed one. The review adds the
  questions such a reading has to answer, and they are the right list: are the arm-level means
  targeted jointly or one at a time; is the targeted mechanism still on the simplex; are the
  armwise tilts variation-independent; what is the *joint* corrected curve and hence the
  covariance a contrast needs; is positivity arm-specific; and does the theorem hold `K` fixed.
  The renormalisation problem above is the second of those, so this is one known defect inside a
  set of unasked questions rather than the only thing in the way — and a simplex-preserving
  parameterisation, most likely a multinomial fluctuation, is the shape of the answer.

**The order to work in**, revised, and it follows from what blocks what rather than from
effort. Piece **0** was first and has landed; what is left is:

1. **A2**, because B's and C's evidence is read *through* the curve it checks, and because
   nothing about it is blocked. Item 20 raises its priority: a component-by-component
   comparison is how a curve built at arrays other than the ones the equations were solved at
   gets found.
2. **B**, because poor overlap may be where the demonstration has to happen and it is currently
   the one regime where the equations are not solved at all — and because item 20 is the
   closing pass's exit, which is this piece's subject.
3. **C**, which is the point.
4. **A1** whenever the paper can be got — out of sequence by necessity, not by choice, and it
   outranks everything above it the moment it is possible. If the theorem and the transcription
   disagree, work above it that has already landed is work to redo.

Then the applied stress tests, which are generalisation checks and not substitutes for C: a
Super Learner library, higher-dimensional and nonlinear processes, moderate near-positivity,
and binary as well as continuous outcomes. A **fitted** weighted run belongs on that list too:
item 17 closed the *transport* on the exact law, which is the right instrument for an identity
about conditional expectations, and is not the same thing as having run one. `repeats=` is off
the list — item 18 ran it. **D** is independent of all of it and should not queue behind any
of it.

### What each new test has to be watched to fail

[Lesson 4](#what-the-sizings-got-wrong) is that a test written after a change and never watched
to fail pins nothing, and [lesson 2](#what-the-sizings-got-wrong) is that this variant's
instruments go blind in a place that can be named in advance. So the mutation goes in the plan
rather than being found afterwards. The review supplies most of this table; the right-hand
column is what makes it usable.

| layer | what it pins | the mutation it must fail against |
| --- | --- | --- |
| unit | `Q_r`, `g_{r,1}`, `g_{r,2}` are the three definitions | swap `gr1` and `gr2` |
| unit | the corrected curve is a **difference** | add the corrections instead of subtracting |
| unit | the curve reads *starred* nuisances | read the initial `g` or the initial reductions |
| unit | arm indexing | swap the arm columns |
| unit | each of equations (8), (9), (10) is solved | drop one equation at a time |
| unit | the stopping rule accepts either ruler | delete the absolute branch (already done, item 12) |
| unit | a weighted fit transports (item 17) | reductions taken at the sampling law (already done, item 17) |
| oracle | the drift decomposition | delete one correction term |
| cross-language | `drtmle` parity, component by component | perturb one component and watch only that row move |
| integration | `guard=()` is `TMLE` bit for bit | route the empty guard through the reduction loop |
| integration | each guard removes its own direction | cross the guard semantics |
| integration | a failing score check is visible in `summary()` (item 16) | silence the verdict (already done, item 16) |
| simulation | slow `Q̄`, wrong `g` | `TMLE` must under-cover, or the regime was not entered |
| simulation | slow `g`, wrong `Q̄` | as above, in the mirror cell |
| simulation | both nuisances right | no material efficiency loss from the corrections |
| simulation | both wrong | no false robustness claim |
| simulation | `√n · R_remaining → 0` (item 13) | freeze the reductions at their initial fit |
| stress | weak overlap | inference must be marked invalid where the scores fail |
| stress | repeated splits (item 18) | reuse one draw's reductions in every draw; drop a draw from the average (already done, item 18) |

Two of these cannot be written against the exact law and the reason is derivable rather than
empirical: everything the variant adds vanishes there row by row. The corrected-curve rows want
nuisances that are wrong on purpose, which is what `tests/unit/test_remainder_drtmle.py` and
`tests/unit/test_influence_drtmle.py` already do and what any new module here has to do too.

### Limitations, recorded rather than fixed

Real, understood, and worth writing down rather than fixing. None of them would change a
coverage number, and each is stated where the code that has it lives as well as here.

**4. The alternation does not reliably converge, and the reason is structural.** Equation (10)'s
covariate is `gr2/gr1`, and `gr2` vanishes exactly where the mechanism is right — so on the fits
anybody actually wants that covariate is nearly zero and its Newton solve is near-singular:
observed at `mean|h| = 1e-3`, `|epsilon|` reaching 280 and a singular Hessian in a third of the
rounds on one unseeded draw. Such a fit runs to the outer cap and reports
`failure = "max_iter_reached"`. `ReductionFluctuation.ill_conditioned` reports it, and `drtmle`
sidesteps the whole question by capping at three iterations and never claiming convergence.

Over [96 fits](#how-the-alternation-exits) the conditioning is **worst where the reasoning
predicted**, which is the part that had never been tested: `gr2` vanishes where the mechanism is
*right*, so the easy process should be the ill-conditioned one, and it is. `linear` reports an
ill-conditioned solve on **5 of 12** fits at `n = 600` and **9 of 12** at `n = 1,200`, against
**0 of 12** for `nonlinear` at `n = 600`. Running out of rounds is a minority — 8 of 96 — but
converging is rarer still: 2 of 96 reached the tolerance and 86 stalled.

**5. Equation (9) is never solved exactly.** Its covariate `Qr/g*` reads the very mechanism it
tilts, so one solve zeroes the score at the pre-tilt covariate and leaves a residual at the
post-tilt one. The closing pass iterates it — to `4e-12` on the exact law and about `1e-9` on a
fitted one — and does not remove it. Equations (8) and (10) *are* exact, so this is the only
term keeping the reported curve's mean off machine zero. **The `1e-9` is the three-quarters of
draws where it behaves**; on the rest it is four to five orders worse and this stops being a
limitation, which is [item 20](#b-the-loops-exit-and-whether-what-it-leaves-is-what-gets-reported).
Whether the two are the same mechanism failing by degree or two different things is not known,
and it is the first question A2 should settle.

**6. The closing pass's mechanism stage stops on its cap, not on its tolerance.** It settles
around `1e-9` rather than reaching `spec.tol = 1e-10`, on **94 of 96** swept fits. Harmless *at
that size* — the steps are arithmetic, and item 5 is why it cannot get there — but a cap that
always binds is worth knowing about rather than reading as convergence, and
[item 20](#b-the-loops-exit-and-whether-what-it-leaves-is-what-gets-reported) is the reason the
qualification now has to be there: the stage this cap binds on is the *mechanism* stage, and
`D*_g` is the term item 20 finds uncentred. A cap that always binds and a term that is
sometimes wrong by `2e-04` are close enough together to be one story, and nothing here has
checked whether they are. The two fits that stopped otherwise are both `weak-overlap`.

**8. `retarget` is no longer arithmetic on cached arrays.** The reductions are refitted inside
the alternation, so a truncation curve or an MNAR sweep costs about a fit per point rather than a
fraction of one, and a result read back from disk cannot retarget at all — its estimator is gone
and there are no learners to refit with. The *records* do survive as of format version 10 —
before it `ReductionFluctuation` was not serialised, which cost a reloaded fit two of its three
score-check rows and is written up in [piece 0](#closed-since-this-list-opened). This is a cost of
following the source, which states equations (9) and (10) at *starred* reduced regressions;
holding them at their initial fit would solve a different equation.

**9. `gr2`'s truncation is fixed at fit time**, so the part of a truncation curve that comes from
equation (10) is flat by construction. `fit_reduced`'s docstring sets out why — the array *is* a
regression of a quotient by the mechanism, so it cannot be stored raw and re-truncated — and why
flat-by-construction reads as insensitivity rather than as a limitation.

**10. Scope is narrower than the source's software**, deliberately and by name: `att`/`atc`, the
other four parameter axes, `delta=`, `intermediate=`, fold-wise targeting and composition with
`CTMLE` are refusals with reasons, listed under [Not written
yet](methodology.md#not-written-yet). `reduction="bivariate"` and a multi-valued treatment are
the two candidates, and they are [piece D](#d-widen-the-scope-to-what-the-sources-derive).

### Closed since this list opened

Kept rather than deleted, because the numbering is frozen: these items are cited by number
from `benchmarks/bench_drtmle.py`, `.github/workflows/drtmle-convergence.yml`,
`estimators/targeting.py` and `tests/unit/test_drtmle_fit.py`, so a closed item's number is
not available for reuse. Items 14, 16, 17 and 18 were **piece 0**, which is why its section is
gone from the list above: none of the four was research, all four were claims the package made
that were wider than the evidence behind them, and what they protected a user from was being
told something the fit had not earned while everything else here is open.

**7. The relative-score exit criterion was a poor instrument — replaced.** The loop exited on
`|score| / mean|h|` against `spec.tol = 1e-10`, and `mean|h|` is `1e-3` to `1e-2` for equation
(10)'s covariate, so an absolutely negligible score read as a large relative one: on **68 of 96**
fits equation (10)'s relative score was above the tolerance while the worst absolute score was
under `1e-3` of `se/√n`. `targeting._solved` now accepts an equation on *either* ruler — the
relative test as before, or an absolute score under `_NEGLIGIBLE / n`, which is the bar
`score_check` already applies to the fit that gets reported. Asymptotic linearity asks for
`P_n D = o(n^-1/2)`; machine zero was never the requirement. It applies to **all three**
equations rather than to equation (10) alone, which was measured rather than assumed: on a
400-row `linear` fit the round the loop gave up at had equation (10) at `2.3e-8` *and* equation
(9) at `3.9e-8`, with the joint likelihood flat to six decimals — the two trade off, so relaxing
either alone stops nothing. Equation (8), whose `1/g` is bounded below by the truncation, still
stops on the relative test, so a well-conditioned fit is unaffected. Refitting three processes at
two seeds under both rules, every fit moved from `stall` to `tolerance` and took a third to a
tenth of the rounds — `linear` 30 → 3, `nonlinear` 22 → 8, `weak-overlap` 36 → 11 — while the
worst score `score_check` sees was no worse and usually better, and `ate` moved by at most
`4.1e-5`, which is `2.4e-4` of a standard error. What loosened is the *loop's* internal stopping
rule and nothing a reader is shown: `score_check` still holds the reported fit to `1e-3·se/√n`,
which is why it still fails 23 of 24 `weak-overlap` fits.

**14. Validity is not efficiency, and now the output says so.** The prose never called the
corrected curve efficient — checked, and the review's charge came back narrower than stated.
The *output* did: `score_check` signed a `DRTMLE` fit off with "the targeting step solved the
estimated **efficient** score equation" over three rows, two of which are the corrections, and
`inference/influence.py`'s first line called every curve in the module efficient. A corrected
fit's verdict now names what it solved and says the curve is `D = D* − D*_Q − D*_g`, valid
under weaker conditions rather than efficient under them; the paragraph is in the appendix, the
guide, `estimators/drtmle.py` and a paragraph of `reduced_corrections`, and the README cell says
it in a clause. A *plain* fit's verdict is byte-for-byte what it was — `README.md`'s transcript
quotes it — which `tests/unit/test_drtmle_fit.py` pins in both directions, watched to fail
against deleting the branch and against never setting the flag.

**16. A failing score check is now on the face of the report.** `score_check` was opt-in and
`estimators/drtmle.py` told the reader to run it "on every fit rather than assuming" —
documentation standing in for reporting, on an estimator whose only product is inference.
`TMLEResult.score_verdict` carries the verdict, derived from the fluctuations rather than
stored, and `summary()` ends with it whenever the check fails. **On every fit, not only a
doubly-robust one**, which is wider than this item proposed: the argument does not depend on
which estimator left the score unsolved. And **only** when it fails, which is narrower: a
passing fit prints nothing new, so every transcript in the README and the guide is unchanged
and the line is worth reading when it does appear. The interval is still printed — saying it is
unlicensed is this item, predeclaring which regimes are refused outright is
[piece B](#b-the-loops-exit-and-whether-what-it-leaves-is-what-gets-reported)'s and needs its
evidence. Serialising the records the check reads was a prerequisite and is format version 10;
see the note under item 20.

**17. The weights claim is the narrow one now, and it is checked.** The docstring said
`weights=` "needs nothing said about it: the reduced regressions are fitted by weighted loss
and every score equation here is weighted". Both halves are true of the code and neither is the
claim that needed making — the derivation was read at an unweighted law. It now says what
transports (the reductions are `P_w`-conditional expectations because they are fitted by
weighted loss, and the mechanism they condition on and divide by is the `P_w` one because
`nuisance.propensity` *is* the weighted fit) and where it stops (an estimated weight).
`tests/unit/test_remainder_drtmle.py` takes the whole expansion at two tilted laws and keeps
the wrong transport as a test: reductions at the sampling law leave a first-order remainder a
single guard no longer removes. Running that under *both* weight functions is what found the
blind spot now written into the module — a weight reading `W` alone is a covariate shift, so it
leaves every conditional alone and the mutation is a no-op there. A **fitted** weighted run is
still open and is an applied stress test, not this.

**18. `repeats=` averages what it says it averages, and checking it found item 20.** Each draw
runs its own alternation against its own refitted reductions — pinned by the two draws' `Qr`
differing and by `score_check` reporting three solved equations per draw — the report is the
mean of the draws, and no draw is dropped. The mutation this list proposed, "drop a repeat and
watch the averaged curve decentre", is **inert**: a centred curve carries its own `−ψ_r`, so
the mean of any subset of centred curves is centred. What a dropped draw moves is `psi` and the
row count, and the tests bite on those. What it did surface is item 20, which is a defect in
the *fit* rather than in this keyword.

**12. That change is now pinned by a test.** It was not, for a while, and the gap is the one
`CLAUDE.md` names: the whole 61-test `drtmle` suite passed identically before and after, because
every assertion in it is about the *reported* fit and the closing pass makes that fit the same
either way. `TestAnEquationStopsOnEitherRuler` in `tests/unit/test_drtmle_fit.py` unit-tests
`_solved` directly, and the absolute branch was deleted and the suite watched to fail — two of
the four assertions go red — before the test was kept. Asserting `exit_reason == "tolerance"` on
a fitted result was rejected: which exit fires is a property of the draw. The item's two other
loose ends are open and are in [piece B](#b-the-loops-exit-and-whether-what-it-leaves-is-what-gets-reported).

### How the alternation exits

96 fits: four processes by two sizes by twelve seeds, `glm` on both nuisances, `n_folds=5`,
`learner_folds=3`, both the data seed and the fold seed varying. Dispatched as
`.github/workflows/drtmle-convergence.yml` from `benchmarks/bench_drtmle.py`, 2,588s of runner at
42.6s per fit, and **no fit raised**. The rows are kept here for the reason `bench_tmle.py` keeps
its own: a comparison nobody can rerun becomes folklore.

**These numbers measure the exit criterion item 7 replaced**, not the current one. That is
deliberate and is the order the item required — the failure had to be characterised before the
threshold moved — and re-measuring under the current rule is part of [piece
B](#b-the-loops-exit-and-whether-what-it-leaves-is-what-gets-reported).

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
— the prediction item 4 makes and had never tested. Then `check fails`, which is flat zero
everywhere except `weak-overlap` and the one `linear` draw at `n = 600`, and is the item piece B
is about.

## Refusals worth lifting

Parameters this package already had the machinery for and had simply not written down, drawn from
[Not written yet](methodology.md#not-written-yet). Each answered a question applied causal
inference actually asks *and* rested on a derivation that was already settled, so the work was
transcription and checking rather than research.

**All six have landed, and this list is now empty.** It is kept because the numbering is cited
from `tests/unit/test_causal_data.py` and `tests/unit/test_influence_gateaux_shift.py`, and
because the order was a **dependency order**: item 1 unblocked item 6, whose whole content is the
contrast machinery item 1 built; item 2 built the projection machinery item 4 copied; items 3 and
4 both changed `fit_regimen` and `fit_mechanism`, so doing them adjacent was one round of churn in
those signatures rather than two — and it paid, since item 3 left the recursion carrying the
data's weights and item 4 inherited them rather than adding them.

| | what it lifted | where it is documented |
| --- | --- | --- |
| 1 | `att[a vs ref]` and `atc[a vs ref]` for a **multi-valued treatment** — one contrast per non-reference arm, a column per contrast rather than a new fluctuation group, opt-in on a multi-arm fit via `default_arms="binary"` | [multi-valued treatment](user-guide.md#multi-valued-treatment) |
| 2 | a **non-identity link** for `msm=` — `link="log"` and `"logit"` make a coefficient a log risk ratio or a log odds ratio, and `res.coefficients(scale="ratio")` exponentiates them | [the MSM section](user-guide.md#summarising-the-arms-a-marginal-structural-model) |
| 3 | **observation weights for `LTMLE`** — the declared regimen parameters at `dP_w = w dP / E[w]`, with every node's nuisance fitted by weighted loss and every node's score equation weighted | [treatment over time](user-guide.md#treatment-given-over-time) |
| 4 | a **working model over regimens** for `LTMLE` — `msm_regimen[<term>]`, the `h`-weighted projection of `E[Y^ā \| V]`, under every link, with the horizon inside the design on a survival fit | [summarising the regimens](user-guide.md#summarising-the-regimens-a-marginal-structural-model) |
| 5 | **`delta=`, `intermediate=` and `weights=` on a `shifts=` fit** — all three were refused together on a reason that was wrong for all three, so lifting them was one change rather than three | [missing outcomes, an intermediate, and weights on a dose](user-guide.md#missing-outcomes-an-intermediate-and-weights-on-a-dose) |
| 6 | the **omitted-variable bound, the E-value and the MNAR tilt at `K` arms** — one analysis per contrast, since `cf_d` is the share of the *Riesz representer's* second moment a confounder would add and a representer belongs to one linear functional | [multi-valued treatment](user-guide.md#multi-valued-treatment) |

Item 5 left a narrower gap behind than the one it replaced: the **MNAR tilt on a shift fit** is
still refused, because the tilt re-mixes `Q̄` under a moved mechanism, a shift's plug-in is `Q̄` at
the assigned dose, and whether the tilted parameter is still the shift parameter has not been
derived. That is a missing derivation rather than missing transcription, which is why it is not
carried forward as an item of its own.

## What the sizings got wrong

Seven lessons, distilled from the per-item retrospectives that used to run to several hundred lines
on this page. They are kept and the retrospectives are not, because the only thing a retrospective
is for is the next sizing — the full pre-work read of what `drtmle` would touch, the per-seam
record of what each cost, and the six landed refusals' own notes are in git history, last carried
in full at `da8cacf`.

**1. A refusal's stated reason is the first thing to check, and it is wrong about half the time.**
Three of the six lifts above found the written reason false rather than merely stale. `shifts=`
refused `delta=`, `intermediate=` and `weights=` together on one reason that was wrong for all
three — conditional probabilities of binary events do not become densities because `A` is
continuous, and a weight tilts the population rather than entering the clever covariate.
`LTMLE`'s weights refusal claimed they "put a further per-unit factor in the clever covariate's
denominator at every node", and they do not. The omitted-variable bound's refusal claimed `cf_d`
was a coefficient in a treatment equation, and it is not. In each case a one-line reason had been
written once and never re-derived, and the lift was smaller than the refusal implied. **Re-derive
the reason before sizing the work.**

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
code, three of [the review](drtmle-review.md)'s charges came back narrower than stated (14, 17
and 19, each still real) and the rest came back whole. That ratio is about the same as lesson
1's on refusals, and for the same reason: a written justification is a claim with no instrument,
so it decays at the rate claims with no instrument decay.
**The cheapest instrument for a prose claim is a reader who has the source open**, and one pass
of that over this page cost less than any item on it.

**7. A test can pin the wrong half of what it is named for, and then it decays like a prose
claim.** Piece 0 turned up two of these and neither was on this list. `TestItSurvivesARoundTrip`
had asserted a `DRTMLE` fit's estimates and curves came back intact for as long as the variant
had existed, and they did — while the *diagnostic* did not: `Fluctuation.mechanism` and
`.reduction` were never serialised, so a reloaded fit's score check answered a strictly weaker
question under the same name, and could pass where the live check failed. Limitation 8 recorded
that as lost record-keeping for two versions, which is what a defect looks like when it has been
written down in the wrong register. And every test in `test_drtmle_fit.py` read one fit on one
split, so item 20 — a quarter of splits leaving the reported curve uncentred — could not be
seen from inside the module until a second split was added for an unrelated reason. Lesson 4
asks whether a test fails when the code is wrong; the two here failed to ask *which* wrong.

There is a third, about this page rather than about the tests, and it is the same mistake one
level up. Item 20 was first written into [limitations recorded rather than
fixed](#limitations-recorded-rather-than-fixed) — a section whose own preamble says none of its
entries would change a coverage number. An uncentred influence curve is a variance estimate for
something the fit did not compute, so it can change one, which makes item 20 a link-4 failure
and a piece's business rather than a limitation's. The entry itself was accurate and
cross-referenced from six places; it was the *heading above it* that told a reader to
discount it. **Where a finding is filed is part of what it says**, and a section preamble is a
claim about every entry under it — so adding an entry is asserting that preamble again.

## On native acceleration

A Rust extension for the numerical kernels was planned. `benchmarks/bench_tmle.py` says it is not
worth building. Profiling a full fit by module (`cProfile`, total time):

| fit | cleverly-authored code | scikit-learn + LightGBM |
| --- | --- | --- |
| n=5,000, `library="default"` | **0.5%** | 44% |
| n=20,000, `library="glm"` | 22% | 17% |

The targeting step is 1.5–1.7% of a `glm` fit and does not appear at all in a `default` one — it
is a 2×2 Newton solve with a closed-form Hessian. Nuisance estimation dominates, and it already
runs in compiled code. Note how much the preset matters: `glm` is the cheapest library available,
so it makes every other line's share look several times larger than it is. Benchmark with
`--library default` before drawing a conclusion.

The 22% figure above is almost entirely *one* function, and profiling it turned up waste rather
than arithmetic — waste that was cheaper to fix than to rewrite:

- **The multiplier bootstrap was 92–95% multiplier *generation* and 2–3% matrix product.** It
  drew a full float64 uniform to produce one Rademacher sign. Generating bits instead is ~2.4×
  faster. Better: for `multiplier_kind="normal"` the max-t law has a closed form — `xi @ IC` is a
  linear map of a Gaussian — so the whole resampling loop collapses to one covariance and a draw
  from an *m*-dimensional normal, which is **80–360× faster** and never allocates a
  `(n_replicates, n)` array.

  That speed is not free, and `multiplier_kind` still defaults to `"rademacher"`. The closed form
  exists *because* the Gaussian max-t law depends on the influence curves only through their
  covariance — so `"normal"` is a plug-in normal approximation rather than a resampling scheme,
  and it cannot see the leverage a `1/g(W)` clever covariate produces under weak overlap.
  Simulated against a brute-force max-t distribution, it is biased conservative there (+0.14 on a
  true 2.16 at n=200, +0.07 at n=2,000), while `"rademacher"` stays within 0.02. On well-behaved
  influence curves all three kinds agree. Use `"normal"` when *n* is large, the curves are well
  behaved, and resampling actually shows up in a profile.
- **The cluster bootstrap rebuilt its membership index inside every replicate**, an
  `O(n_clusters × n)` scan per draw. Building it once is **24–160× cheaper** per replicate, which
  a 1000-replicate cluster bootstrap pays back a thousand times over.
- `cluster_sums` used `np.add.at`, which is unbuffered; `np.bincount` is ~2× faster.

None of that needed Rust, and the package stays pure-Python. The other place that mattered turned
out to be thread scheduling rather than arithmetic: nuisance fits run single-threaded by default
so parallelism happens across folds and candidates instead of inside each fit (see
`cleverly.learners.set_thread_limit`).

**When to revisit this.** Native code pays where the nuisance estimator is *not* a scikit-learn
model, and today none of them is. The trigger is **HAL** (highly adaptive lasso) and its
undersmoothed variant: a zero-order spline basis of `n × O(n·d)` binary indicators that
scikit-learn's lasso cannot take, where basis enumeration, sparse assembly and coordinate descent
are a natural fit for a native extension — R's `hal9001` ships a C++ backend for exactly this. The
EP-learner benefits *through* HAL rather than on its own; its other cost is targeting a
*k*-dimensional score with *k* = basis size, which is BLAS-bound and already fine. Longitudinal
and survival TMLE are weaker cases: the loop over timepoints is Python, but each body is a
nuisance fit, so they stay scikit-learn-bound. That remains a prediction rather than a measurement
— `benchmarks/bench_tmle.py` has no `LTMLE` case, so profile one before acting on it.

The measurement is reproducible — rerun the benchmark before revisiting this.
