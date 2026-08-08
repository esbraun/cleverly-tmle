# Roadmap

What has landed, what is open, and where native acceleration does and does not pay.

**Read [the standing decisions](#standing-decisions) first.** It is one screen, one row per
decision, each with the condition that would reopen it and a link to its evidence. It exists so
that "why is there no `numba` dependency" costs a table rather than seven measurement write-ups.
The table is the verdict; the linked documents are the evidence.

## Where `DRTMLE` stands

`DRTMLE`, the doubly-robust-inference variant, ships under **conditional validity**:
[`docs/drtmle.md`](drtmle.md) is its contract, and it is the page to read rather than this one if
the question is what the estimator computes, what it refuses, or what a caller has to check. In
one paragraph — the algorithm computes what Benkeser et al.'s Theorem 1 derives, checked against
the theorem's appendices, against the Gateaux derivative of the parameter, against exact
finite-support laws and against the remainder identities; and the interval it reports is valid
*conditional on* the practitioner obtaining adequate primary **and reduced-regression** fits,
which are rate conditions no fit can check for itself.

**A closed validation programme is what that claim rests on, and it is retired.** Six lettered
pieces over twenty-six pull requests: a theoretical audit against the sources, a
targeting-and-exit sweep, a controlled coverage demonstration, a reference study for the reduced
regressions, a construction ablation, and a terminal experiment. It established the fidelity and
the numerical validity, it measured a large and reproduced improvement over a plain `TMLE` where
one nuisance is badly estimated, and it **did not** reach nominal coverage or localize why.
[What is still open](#what-is-still-open) is that readout in full.

**Its machinery is out of this tree and recoverable from one tag.** The study harnesses, the
committed per-replicate evidence, the R differential records, the dispatch workflows and the ten
working documents are reachable from **`drtmle-validation-archive-2026-08`** and were removed
from `main` once the programme closed. What replaced roughly twenty-six thousand lines of
investigation is `docs/drtmle.md` plus the readout below: the repository now carries tests of the
mathematics and the software contract rather than tests of the investigation that produced
confidence in them. Two consequences worth stating plainly. A coverage number is **not** a
regression test — it depends on the learner, the DGP, the sample size and the dependency
versions — so the only statistical run left is a broad catastrophic-regression guard in the
nightly tier, and it says so in its own docstring. And a claim about the programme that is not in
`docs/drtmle.md` or below is a claim to check against the tag rather than to repeat.

Everything else here is a record: [Refusals worth lifting](#refusals-worth-lifting) is the list of
parameters this package had the machinery for and had simply not written down, and it is now
empty.

Nothing is queued behind `DRTMLE`. The remaining rows under [Not written
yet](methodology.md#not-written-yet) are there because nobody has asked, not because anything
stands in the way, and the one standing conditional item is the [HAL
trigger](#on-native-acceleration).

## Standing decisions

Choices this package has made, each on evidence rather than on taste, and each of which
someone will otherwise re-derive from scratch. **This table is the verdict; the linked
documents are the evidence.** Read a row before opening what it links to — the point of
having it is that "why is there no `numba` dependency" should not cost seven measurement
write-ups.

Each row also says what would *reopen* it, because a decision with no such condition is a
prejudice.

| decision | why, in one line | reopened by | evidence |
| --- | --- | --- | --- |
| **`numba` is a benchmark-only dependency.** Nothing under `src/` imports it | all three "adopt numba" recommendations dissolved once the numpy baseline was written properly — an expansion that need not happen, a sort already done elsewhere, a quadratic mask rebuild — and the largest single win in the whole investigation was a context manager | a kernel beating a *competent numpy* baseline, on a machine with more than four cores. Nothing is measured above four, and no CI job runs above two | [benchmarks](benchmarks/) |
| **No Rust or other native extension**, and the package stays pure Python | nuisance estimation dominates every realistic fit and already runs in compiled code; cleverly-authored arithmetic does not reach 3% of a `default` fit even at five million rows | **HAL** — a nuisance learner that is not scikit-learn moves the whole denominator. Unchanged, and still the trigger | [On native acceleration](#on-native-acceleration) |
| **The internals are numpy, not polars**, whatever the caller passes in | the whole dataframe boundary is 1.5% of the cheapest fit and 0.06% of a realistic one — 1.5% and 0.04% asymptotically — so there is no share for a columnar engine to win, and scikit-learn takes contiguous numpy regardless | a workload whose cost is joins, group-bys or IO. None of those is on this path | [At several million rows](#at-several-million-rows) |
| **No R under `src/`, in `noxfile.py`, in any test tier, in the fast CI — and no parity test anywhere** | two checks that cannot fail against the same class of error are one check. Both packages descend from one source, so agreement with R is evidence about a *transcription* and can never certify a derivation — and it is blind in exactly the place an exact-law check is, since at correct nuisances the corrections vanish row by row. The worked example is the sign of the mechanism correction, which a parity run would have recorded as correct and been right by luck; reading the paper's appendices is what settled it | **it was reopened once**, for a bounded differential *diagnostic* against the published R package, and that instrument has since been run, read out and retired with the rest of the validation programme — its records are at the `drtmle-validation-archive-2026-08` tag. Reopening it again needs a question the theorem, the exact-law identities and the archived records cannot adjudicate, named in advance, whose two possible answers lead to different already-specified next steps. What no such run may ever do is justify a production change, which is [stop-ship 17](#stop-ship) | [the sign of the mechanism correction](drtmle.md#the-sign-of-the-mechanism-correction); [lesson 9](#what-the-sizings-got-wrong) |
| **A study that selects and then certifies runs on two disjoint cohorts of draws**, separated by a commit of the frozen selection | disjoint quadrature blocks split the *integration* noise, and the simulation draw is the independent unit: a rung chosen across a set of draws and certified on the same set is a data-dependent selection assessed on the sample that made it. The disjointness is checked on the **data seed**, since two draws sharing one under different splits are the same rows twice | a study whose selection is not data-dependent — a rung shipped rather than measured is one cohort's work, which is what E2 was | the terminal experiment ran on a selection cohort and a disjoint audit cohort on exactly this rule, and nominated nothing on either; the record is at the `drtmle-validation-archive-2026-08` tag |
| **A fidelity gate passes on a non-inferiority bound, never on failure to show superiority — and what it then establishes is *relative*, not absolute** | an interval containing zero establishes neither equality nor adequate approximation, so a gate read that way certifies whatever it cannot resolve — and E2R's own record called two rungs "genuinely indistinguishable" on exactly that reading. **The second clause is new and is a correction**: `held_out_risk` differences estimate `‖m − f‖²_w − ‖m − h‖²_w`, so non-inferiority against a finite ladder bounds a risk *difference* and never `‖m − f‖²_w` itself. Every candidate can share a material approximation bias while their gaps stay small, and rejecting a coarse negative control proves discrimination against *that control*. Gate B **ranks**; it does not certify adequacy, and **F6 either anchors it or renames it** — this row said F5, which is the row that *reads* a ranking rather than the row that repairs the gate, and the same absent anchor is why F5's tier-2 ceiling arm is reported as an estimate rather than as an oracle | a margin that cannot be tied to a tolerable change in the reported column. The two composites' is `(δ/3)²/(n·weight_scale)` by Cauchy–Schwarz; the three componentwise ones have no such transfer and take a share of the negative control's measured distance instead | the gate was never anchored to an absolute risk, and the piece that would have anchored or renamed it did not run; a future study reusing this shape has to settle that first |
| **Nuisance fits run single-threaded**, with one `ThreadpoolController` per process | parallelism belongs across folds and candidates rather than inside each small fit; building the controller per entry was 57% of a DR-TMLE `retarget` | a fit large enough that one model wants the machine — `set_thread_limit(None)` is the lever, not a code change | [`thread_limit_profile.md`](benchmarks/thread_limit_profile.md) |
| **`tracemalloc` is the memory instrument** | it *does* see numba's allocations, through all three CPython allocator domains — the caveat that said otherwise was wrong and was measured to be wrong | a question about resident memory rather than allocation, or a library calling `malloc` directly. That needs an incremental-RSS arm *beside* this column, not instead of it | [`production_plan.md`](benchmarks/production_plan.md) §1.3 |
| **Benchmark write-ups live in [`docs/benchmarks/`](benchmarks/)**; `benchmarks/results/` is generated output and is git-ignored | a `results.jsonl` from a four-core container reads as a fact about the package rather than about that box | nothing | [`docs/README.md`](README.md) |
| **A dispatched *study*'s per-replicate rows are manifested, and archived rather than left in CI.** The row above is about generated benchmark output and this is its exception, not a contradiction of it | a CI artefact has a 90-day retention, and C3c's four are the only copy of the 6,000 fits every gate verdict on this page is read from. A summary table is not the evidence, it is a transcription of it | nothing. The cost is a manifest per study and a few megabytes | `drtmle-validation-archive-2026-08` |

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
  finished, and calling it landed would be claiming the part that is missing. **One defect is
  live and it is a stop-ship; the other is closed and is worth keeping visible:**

  - ~~the mechanism correction's sign disagrees with Theorem 1~~ (item 21) — **closed, and the
    curve agrees with the theorem.** The working paper is now in the repository
    (UCB Biostatistics paper 356, p. 9) and read. Its §3.1 display does define `D_A = −(Q_r/g)(A − g)`
    while Theorem 1 subtracts `D_A`, which read together flips the mechanism correction — but the
    paper's **own appendices** derive each block in a form satisfiable only with the *positive*
    correction, and Theorem 1's variance formula then reads exactly as this package computes it.
    So the display is wrong on its face, in a document that also prints `D_Y` twice with two
    signs. The argument, the two further slips in the same paper, and what is left are in [the
    concordance's §4](drtmle.md#the-sign-of-the-mechanism-correction);
    `tests/unit/test_theorem_drtmle.py` pins it, watched to fail against a flipped sign in the
    library itself;
  - **the reported curve is not centred whenever the targeted mechanism leaves the truncation
    bounds** — on roughly a quarter of ordinary splits and on 23 of 24 weak-overlap fits, at
    `2e-05` to `7e-04` where a solved fit sits near `1e-09`, while the loop's own three rows all
    report their scores solved to `1e-11`. That is items 20 and 11, and they are **one defect
    with a located cause**: equation (9) is solved against the *raw* tilted `g*` and the `D*_g`
    the curve subtracts reads the *truncated* one, so a single clipped row of 600 is enough to
    decentre the curve while every fluctuation row still reports `1e-11`. It is a `DRTMLE`-only
    defect and it is not confined to poor overlap. The investigation
    log carries the measurements.

  **B1b has landed and closed it.**
  `solve_bounded_mechanism` solves equation (9)'s score at the truncated tilt — the expression the
  curve carries — and the alternation carries that truncated array forward, so the two are one
  expression at one state. Measured on all four fixtures the defect was characterised on,
  including `weak_overlap` at a forced `g_bounds=(0.15, 0.85)` where 375 rows clipped: every
  identity at `1e-17` or better, every final score `1e-10` against a bar near `5e-06`, every score
  check passing. A fit whose bound never binds is bit for bit what it was.
  It was caught *by name* first, which is why the fix is checkable rather than asserted:
  B1a made `res.validation.correction_check()` recompute
  each arm's `Pn[w D*_g]` and `Pn[w D*_Q]` from the exact returned state and report the residual
  against the score the loop recorded, and no threshold in it was loosened to make those rows
  pass. Before it, the only witness was the influence-curve rows being uncentred, which was how it
  was found at all. Item 21 was caught by nothing here and
  could not have been — it took the source, and the source only settled it because its
  appendices could be checked against arithmetic this repository already had.

  Beyond those: **the influence curve was transcribed from R's `drtmle` rather than derived, and
  that is now its provenance rather than its standing** — item 1, [closed by
  A1a](#closed-since-this-list-opened). Its evidence is two independent checks against the
  derivation: Theorem 1's appendices at a nonzero `Q_r`, and the Gateaux derivative of the
  parameter on a law where one nuisance is wrong on purpose. Nothing here
  has been compared against that package's numbers and **nothing will be**, which is a
  [decision](#closed-since-this-list-opened) rather than a gap; and a coverage study found **no gap
  for the variant to close** at the sizes it could reach. [What is still open](#what-is-still-open) is the
  rest. Three of its items are there because the first review put them there:
  the theorem's *other* assumption beyond the three score equations (item 13), the cross-fitting
  construction the reductions would need to satisfy it (item 15), and the claim that the
  alternation converges (19). Four more of that review's items — that the corrected curve is the
  efficient one (14), that a fit whose score check fails still reports an ordinary Wald interval
  (16), that weights need nothing said about them (17), and that `repeats=` averages what it is
  averaging (18) — were piece 0 and have [landed](#closed-since-this-list-opened)

## What is still open

`DRTMLE` ships under **conditional validity**, and
[`docs/drtmle.md`](drtmle.md) is the contract: the algorithm computes what Benkeser et al.'s
Theorem 1 derives, and the interval it reports is valid *conditional on* the practitioner
obtaining adequate primary and reduced-regression fits. Those are rate conditions on five
estimated functions. They are not verifiable from a fit's own output, and **numerical score
convergence does not verify them** — which is the single most important thing a user of this
variant has to know, and is why it is on the face of the estimator's own docstring.

### What the validation programme established, and what it did not

A closed programme of six lettered pieces — a theoretical audit against the sources, a
targeting-and-exit sweep, a controlled coverage demonstration, a reference study for the reduced
regressions, a construction ablation, and a terminal experiment — ran across twenty-six pull
requests and is now retired. Its harnesses, its per-replicate evidence, its dispatch records and
its working documents are reachable from the **`drtmle-validation-archive-2026-08`** tag and are
not in this tree. What it decided is in [the standing decisions](#standing-decisions); what it
established is here.

**Established.**

- *Theorem fidelity.* The corrected curve is the Gateaux derivative of the parameter, checked on
  laws where one nuisance is wrong on purpose — which is the only place it is checkable, since at
  correct nuisances `Q_r` and `g_{r,2}` vanish row by row and every exact-law check goes blind.
  The sign of the mechanism correction is the appendices' orientation and not the §3.1 display's.
  The reported variance is Theorem 1's own `P_n{D* − D_A − D_Y}²`, and the contrast reads the
  covariance rather than the sum of the arms.
- *Numerical validity.* Each recorded score is an identity against a recomputation of the term the
  curve carries, per arm and per equation, on the face of every fit — and the identity **holds**,
  including where the mechanism truncation binds, which it did not before the constrained
  estimating equation landed.
- *Inferential usefulness, in part.* The interval is materially better than a plain `TMLE`'s where
  one nuisance is badly estimated: `0.844`/`0.848` against `0.532`/`0.472` at `n = 2,400` in the
  cell built for it, a paired `+0.312` and `+0.376`, reproduced over two independent seed batches.

**Not established, and recorded as such rather than as a defect.**

- *Nominal coverage, anywhere.* The best reading in a 6,000-fit study was `0.880`.
- *A localized cause.* A six-contrast construction ablation over 2,496 fits returned a **null** on
  its primary column, and a terminal experiment run on a selection cohort and an independent audit
  cohort **nominated nothing**. The failure is measured and was never localized.
- *The remainder condition.* `√n R_remaining` is flat rather than falling at reachable sizes, and
  Theorem 1's `R_{Q,n}` and `R_{g,n}` were never bounded separately.
- *Any `src/` change justified against the theorem.* None was ever licensed, and none was made.

**Two measured shortfalls are one premise measured twice**, which is worth stating because three
revisions of this page read them as two mechanisms. The second-order remainder Theorem 1 assumes
negligible does not vanish at these sizes, and the reported `se` runs about 10% short of the
spread it covers in `q-drift` — and about 16% *long* in `g-drift`, which is what costs the
"defect in the variance estimator" reading its mechanism. `σ̂²_n` is valid to first order exactly
when the condition the first quantity fails holds.

### What would reopen it

The regime the variant is *for* is an **adaptive** good nuisance converging more slowly than
`n^(−1/4)`, at an `n` large enough for the coverage decay to show. That was out of reach of every
budget this programme had, and it is the shape of the run that would say something new. Two
conditions on any such run, both learned the expensive way:

- it must fit the **reduced** regressions with a learner whose consistency is argued rather than
  assumed. Every coverage number on record was taken at `glm` reductions, so it measures a
  *configuration* and not the theorem's premise — and
  `tests/unit/test_oracle_reductions.py` is the standing demonstration that wrong reductions move
  the estimate while every score equation still passes;
- it must select on one cohort and certify on a disjoint one, and say so before it runs.

Nothing about that is blocked. What it is not is a gate on the current release claim, which is
conditional and states its conditions.

### The four links, and why none implies the others

Kept because it is the frame the whole programme was read under, and it is still the right one for
anyone extending the variant: a curve that matches the theorem proves nothing about whether the
alternation solved its equations on a given draw; a solved fit proves nothing about whether the
interval is ever *better* than the one `TMLE` already reports; and a coverage number proves nothing
about which of the two the estimator got right.

1. **Theorem fidelity** — the equations solved and the curve reported are the ones the derivation
   gives, under conditions the fit actually meets.
2. **Derivation-anchored correctness, component by component** — each object the curve is built
   from agrees with what the derivation gives for it, not merely `psi` and `se`, where several
   differences cancel. **The exact law is not enough by itself**, and that is the trap worth
   carrying forward: at the truth the corrections vanish, so agreement with an exact law is blind
   to a flipped sign in the same place a parity run is. What is not blind is a comparison against
   the theorem's terms at a **nonzero** `Q_r`, or against the Gateaux derivative on a law where one
   nuisance is wrong on purpose.
3. **Numerical validity** — every required score solved to a statistically negligible order, *at
   the arrays the reported curve is built from*, and a fit that fails to say so somewhere a reader
   cannot miss.
4. **Inferential usefulness** — coverage in a regime where the plain interval fails.

**There were five links, and the one that has gone was `drtmle` parity**, retired by decision:
both packages descend from one source, so agreement is evidence about a transcription and is blind
to exactly the class of error the sign question was in. The corollary that cost the most to learn
is that *the blind spot is shared with the exact law* — two checks that cannot fail against the
same class of error are one check, however different their machinery.

### Stop-ship

**What these block is the *unconditional* claim, not the release.** `DRTMLE` ships under
[conditional validity](drtmle.md), which states its conditions rather than asserting them; any one
of the clauses below blocks calling the variant finished in the stronger sense — an interval
demonstrated to attain nominal coverage where a plain `TMLE`'s does not. They are the four links
restated as things a reader could check rather than as claims.

**Eight of the seventeen are about how the *evidence* is described rather than about the code** —
10 through 17 — and they exist because a claim that outruns its instrument is how this variant went
wrong repeatedly. Four of those eight (14, 15, 16, 17) are the same shape: **a check that agreed
where it could not have disagreed, described as though it had.** That shape is the single most
useful thing on this page for anyone extending the variant, and it is why the clauses are kept
verbatim rather than compressed now that the programme is closed.

**Clauses 4 through 7 are live on measurements rather than open for want of one**, on 6,000 fits
over two seed batches; clauses 8 and 9 were checked and are not live. A list of blockers with
numbers attached is a better state than it sounds.

**They do not all trace back to one quantity, and two of the four share one.** Clause 6 — coverage
— is the one the remainder drives. **Clause 5 is the `se ratio`, and it has no mechanism of its
own**: an earlier revision named the variance estimator, and `σ²ₙ` is Theorem 1's own formula
rather than a defect. It is the same unestablished premise as clause 6 read through the second
moment, and `g-drift`'s `1.157` — the other side of one — is why it cannot be an omitted variance
term. Clause 4's second half is `cancel` at `1.99x`, which is the branch decomposition failing to
*separate* the two remainder terms rather than either failing to vanish. Clause 3 is 99 invalid
fits, which is the solver. **Four clauses, three mechanisms.** A single-cause reading would send
work to the wrong place; so would inventing a cause per clause.

1. a correction term disagrees with Theorem 1 or the appendix — this was item 21 and it is
   **closed**: the appendices force the positive reading, which is the one implemented, and
   `tests/unit/test_theorem_drtmle.py` is what a reader checks that against rather than the
   §3.1 display;
2. the algorithm's fixed point is not shown to satisfy the theorem's three score conditions —
   item 22, whose *theoretical* half is closed (the paper's step 7 states its own exit as those
   three empirical means, so the order is not prescriptive) and whose numerical half is now
   B2's: whether the paper's order and this one
   reach the same fixed point on real data, both orders run here against the same nuisances. It
   used to say the numerical half was the parity piece's, which was never right — a second
   implementation reaching a third fixed point would have answered a different question — and
   then A1's, which was right about the subject and wrong about the evidence: it is a second
   alternation over a sweep of draws, which is B2's dispatch and nothing A1a could share with.
   **The second alternation exists** — `DRTMLE(update_order="paper")`, landed with
   B2a — and B2b has read
   it over a sweep. **It stays on this list, and the reason is one clause.** Of the rule frozen
   before the dispatch,
   clauses 1, 3 and 4 hold on both processes — the route difference shrinks with `n`, the `se`
   ratios sit at `0.998` and `1.002` at the largest size, and no fit in either route fails its
   score check or its state identity. **Clause 2 fails on both processes at both seed counts** —
   the count of draws where the route difference exceeds a *fold split's* is 2 of 12 and 7 of 36 on
   `nonlinear`, 3 of 12 and 12 of 36 on `weak-overlap`, all **short of half rather than over**,
   which is the route moving `ψ` *less* than a refit of one route does. **And clause 1 is met on
   one process at each seed count, a different one each time**, which says the median it reads is
   underpowered at twelve draws and at thirty-six. Every miss points the way that supports the
   routes agreeing and none points the other; what is stable across all twelve cells is that the
   route difference sits *below* the fold-split difference, by a factor of 3.5 to 4 on `nonlinear`.
   That is a state to describe rather than to legislate away, so the item stays open with its
   evidence and the
   log
   records the two restatements a future revision could make **before** a further dispatch and not
   after this one;
3. a stored score and the term the curve carries are not the same functional of the same state —
   this was item 20, the one that was true and unnoticed, and it is **closed**:
   B1a made it reported and
   B1b made it false, by solving the score at
   the truncated tilt the curve reads. It stays on this list as a thing a reader can check rather
   than as an open item, and the check is `res.validation.correction_check()` on any fit —
   including one whose bound binds, which is the case the numbers used to come from;
4. a required score is not negligible under the predeclared validity rule — **C3c
   read this and it is live**: `1–7%` of `DRTMLE` fits are invalid at tolerance `1e-3` in every
   cell of the study, `TMLE` none, and the rate falls with `n`;
5. `√n · R_remaining` does not trend to zero in either off-diagonal cell, or does so only because
   the two appendix branches cancel — **live, and it is the one the study turned on**. Flat at
   `1.43 / 1.26 / 1.25` in `q-drift` and not falling in `g-drift`, in both seed batches; and
   `cancel` reaches `1.99x` at `g-drift`'s largest size, so the second half of the clause is live
   there too;
6. coverage fails in either cell under the controlled study — **live**: `DRTMLE` reaches `0.844`
   and `0.848` in `q-drift` and `0.780` and `0.784` in `g-drift` at the largest size, none of them
   compatible with `0.95`. Being far better than `TMLE` is a different clause, and it is gate 2's;
7. the invalid-fit rate exceeds its predeclared threshold in the well-overlapped cells — **live**:
   over the frozen `2%` in ten of the study's twelve cell-runs, `0.008` to `0.072`;
8. the conclusion depends materially on excluding failed fits after the fact — **not live, and it
   was checked rather than assumed**: the study's primary numbers are the intention-to-treat ones
   and the excluded column moves them by `0.007` to `0.060`, never across a verdict;
9. it does not reproduce in the second seed batch — **not live**: every qualitative claim
   reproduced across seeds `20250801` and `20250802`, the `se ratio` and the regime-entry column to
   the digit;
10. any document calls the corrected curve efficient under misspecification (item 14's ground,
    which piece 0 cleared and which prose can re-lose);
11. any document calls the curve theorem-derived on the strength of a *display* rather than of the
    appendices — item 21 closed on the second, and the first says the opposite;
12. an unsupported estimand or treatment structure is accepted without a derivation;
13. a weak-overlap interval is reported as valid where the scores fail;
14. any document reads A1a's agreement as evidence about the **cross-fitting** construction. It is
    silent on item 15 by construction, and [the concordance's
    §8.5](drtmle.md#reduced-regression-cross-fitting) says
    so. This is exactly the shape of the mistake item 2 was retired for: a check agreeing where it
    could not have disagreed.

    **This clause used to give the wrong reason, and A1b found that by building what the reason
    was about.** It said the module runs at *saturated* reductions "where every conditioning cell
    is a singleton". On that law the design takes three values over a thousand rows, so the cells
    are not singletons — and saturation of the *reduction* does not decide it in any case, since
    under a primary learner that learns any reduction learner returns different arrays. What makes
    the module silent is `cross_fit=False` **and** oracle primary learners: one fold has no
    complement to nest inside, and a learner that ignores its training rows returns the same
    function whichever rows it saw. The false reason would have licensed reading a *cross-fitted*
    saturated fit as evidence about fold reuse, so this clause was carrying the error it was
    written to catch. `tests/unit/test_drtmle_crossfit.py::TestADataIndependentPrimaryLearnerMakesTheTwoConstructionsAgree`
    is the corrected statement, asserted and kept as a mutation watched to **pass**;
15. any document reads a **bound-active** fit's numbers as evidence that Theorem 1's expansion
    covers it. B1b made the solved score and the reported curve one expression at one state, which
    is internal coherence; the theorem has one mechanism and it is untruncated
    ([§7](drtmle.md#the-bound-inactive-scope)), so a fit
    on which `ĝ` or `g_{r,1}` sits against a bound is [item
    25](drtmle.md#the-bound-inactive-scope)'s third option — empirically supported, outside the
    theorem — however good its identities and its scores look. This is new with the contract and it
    is 14's shape one level up: 14 is a check that could not have disagreed, and this is a check
    that agreed about something else.
16. any document names a **separate defect in the variance estimator** as the mechanism of the
    `se` shortfall, or repairs the shortfall with an inflation factor, a sandwich or a bootstrap
    applied to the released interval. `σ²ₙ` is Theorem 1's own `Pₙ{D* − D_A − D_Y}²`, pinned by
    `test_theorem_drtmle.py::TestTheReportedVarianceIsTheorem1s`, and it is first-order valid
    exactly when the conditions item 13 measures hold. **This page carried the opposite for three
    revisions** and it is the third entry on this list of the same shape as 14 and 15 — a claim
    that outruns its instrument — with the additional feature that it prescribed a repair. The
    check a reader makes is the `se ratio` column in both cells: `0.903` beside `1.157` is not what
    a missing non-negative term looks like;
17. **any change under `src/` is justified by agreement with another implementation**, or any
    document reports cross-language agreement as a release criterion, a correctness result or
    evidence about a derivation. This clause outlived the instrument that occasioned it: a bounded
    R differential was authorized once, to *localize* a first divergence and hand it to the theorem,
    the exact laws and the remainder identities — which are what decide — and it has since been run
    and retired. The clause is unchanged by that, because it was never about the instrument. Both
    packages descend from one source, so agreement is evidence about a *transcription*; the sign of
    the mechanism correction is the worked example, where a parity run would have recorded R's sign
    as correct and been right by luck. Deliberately of the same shape as 14, 15 and 16: a check that
    agrees where it could not have disagreed, described as though it had.

Note what is **not** on this list any more: a coverage gap over `TMLE` of at least 0.05. That is a
product judgment about whether the variant earns its cost, it has no theorem behind it, and it now
lives in gate 2.

### Limitations, recorded rather than fixed

Real, understood, and worth writing down rather than fixing. None of them would change a
coverage number, and each is stated where the code that has it lives as well as here.

**4. The alternation has no convergence guarantee, and equation (10)'s solve is near-singular by
construction — but the loop now converges on 87 of 96 fits, where it used to on 2.** Equation
(10)'s covariate is `gr2/gr1`, and `gr2` vanishes exactly where the mechanism is right — so on the
fits anybody actually wants that covariate is nearly zero and its Newton solve is near-singular:
observed at `mean|h| = 1e-3`, `|epsilon|` reaching 280 and a singular Hessian in a third of the
rounds on one unseeded draw. Such a fit runs to the outer cap and reports
`failure = "max_iter_reached"`. `ReductionFluctuation.ill_conditioned` reports it, and `drtmle`
sidesteps the whole question by capping at three iterations and never claiming convergence.

**The same 96 draws have now been swept twice**, and the entry above used to be written from the
first: before, 2 fits reached the
tolerance, 86 stalled and 8 ran out of rounds;
after, **87 reach the tolerance, 8
stall and 1 runs out of rounds**, at a median of 4 to 9 rounds against 12 to 24 and a seventh of
the wall clock. **Nothing about the iteration changed between them** — what changed is which ruler
the exit test uses ([item 7](#closed-since-this-list-opened)), so the honest reading is that the
loop was reaching its fixed point all along and being told it had not. What has *not* changed is
that no argument here proves the iterates approach a common zero of the three equations, which is
[item 19](#closed-since-this-list-opened) and is why the diagnostics decide rather than the
argument.

The conditioning survives at a third of the rate and **keeps its shape**, which is the part that
had never been tested: `gr2` vanishes where the mechanism is *right*, so the easy process should be
the ill-conditioned one, and it is. `linear` reports an ill-conditioned solve on **3 of 12** fits
at each size — it was 5 of 12 and 9 of 12 — against **0 of 12** for `nonlinear` at `n = 600`. The
rise with `n` on `linear` did not survive; a near-singular round is now something a fit passes
through rather than something it exits at.

**5. Equation (9) is never solved exactly.** Its covariate `Qr/g*` reads the very mechanism it
tilts, so one solve zeroes the score at the pre-tilt covariate and leaves a residual at the
post-tilt one. The closing pass iterates it — to `4e-12` on the exact law and about `1e-9` on a
fitted one — and does not remove it. Equations (8) and (10) *are* exact, so this is the only term
keeping the reported curve's mean off machine zero. **This limitation is bounded, and it is not
item 20.** The two looked like one story — "four to five orders worse on a quarter of draws" — and
are not: the `1e-9` here is the equation the loop poses, measured at the arrays the loop leaves,
and it stayed `1e-9` on the uncentred draws too. What was `2e-04` on those draws was a *different*
expression of the same arrays, which was item 20 and is
[closed](#closed-since-this-list-opened). **B1b left this exactly where it was**, measured rather
than assumed: the final scores on all four of that piece's fixtures sit at the same `1e-09` to
`1e-10`, because what makes the covariate's direction self-consistent is still the outer loop and
not the solve.

**6. The closing pass's mechanism stage stops on its cap, not on its tolerance.** It settles
around `1e-9` rather than reaching `spec.tol = 1e-10`, on **96 of 96** swept fits at `n ≤ 1,200` —
it was 94 of 96 before B1b, and the two fits that stopped otherwise were both `weak-overlap`.
**It is not quite universal once `n = 2,400` is swept**: 102 of 108 on `nonlinear` and 108 of 108
on `weak-overlap` over three sizes, so every exception is at the largest size, which is what
"harmless at that size" would predict of a residual that shrinks with `n`. Harmless *at
that size* — the steps are arithmetic, and item 5 is why it cannot get there — but a cap that
always binds is worth knowing about rather than reading as convergence. The qualification this
entry used to carry — that a cap always binding and `D*_g` being wrong by `2e-04` are "close
enough together to be one story" — was a guess and has been checked: they are **not** one story.
The stage does bind on the mechanism, and the uncentred draws were the ones where the tilted `g*`
leaves the bounds, but the cap bound on 94 of 96 fits while the curve was uncentred on a quarter of
them, so the cap cannot have been what selected them. This entry used to predict that a
bounded-residual convention would make it
*worse*, on the reasoning that a truncated residual is not the canonical logistic score. B1b
adopted one and four fixtures could not tell; **B2b can,
and the prediction is confirmed in the smallest way it could be: 96 of 96 against 94, on the grid
the 94 was measured on.** The two that used to stop otherwise were both `weak-overlap` and no
longer do. So this is the one place a bounded convention cost something, the cost is that a cap
which nearly always bound now binds on every fit at the sizes the prediction was made about, and
item 5 remains why the stage cannot reach `spec.tol` at all. It is worth stating as a measurement
precisely because it was carried as a guess for two revisions — and the `n = 2,400` exceptions are
worth keeping beside it, because a limitation that weakens with `n` is a different object from one
that does not, and nothing here had looked.

**8. `retarget` is no longer arithmetic on cached arrays.** The reductions are refitted inside the
alternation, so a truncation curve or an MNAR sweep costs about a fit per point rather than a
fraction of one, and a result read back from disk cannot retarget at all — its estimator is gone
and there are no learners to refit with. The *records* do survive as of format version 10 — before
it `ReductionFluctuation` was not serialised, which cost a reloaded fit two of its three
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
yet](methodology.md#not-written-yet). `reduction="bivariate"` and a multi-valued treatment are the
two candidates, and they are piece D.

### Closed since this list opened

Kept rather than deleted, because the numbering is frozen: these items are cited by number from
`estimators/targeting.py` and `tests/unit/test_drtmle_fit.py`, and from the archived study
harnesses at the `drtmle-validation-archive-2026-08` tag, so a closed item's number is not
available for reuse. Items 14, 16, 17 and 18 were **piece 0**, which is why its section is gone
from the list above: none of the four was research, all four were claims the package made that
were wider than the evidence behind them, and what they protected a user from was being told
something the fit had not earned while everything else here is open.

**11 and 20. The reported curve is centred wherever the truncation binds, and the two items were
always one failure.** The full diagnosis stays where it was written, in
piece B2's section, because it is the evidence
B1b was chosen against; what closed it is B1b.
Equation (9) is solved at the **truncated** tilt — the expression the curve subtracts — and the
alternation carries that array forward, so the stored score and the reported term are one
evaluation of one expression at one state. The identity holds at `1e-17` or better and the final
scores at `1e-10` on all four fixtures the defect was characterised on, weak overlap at a forced
bound included.

Three things about the closure are worth keeping and each was a fork:

- **It is a measurement rather than an argument.** B1a's rows, its `1e-12` bar and its per-arm
  recomputation are untouched; what changed is the verdicts. A convention adopted *and* a
  threshold relaxed would have been two changes and no evidence.
- **A fit whose bound never binds is bit for bit what it was**, because the bounded solve returns
  the unconstrained one untouched where the clip is slack on every row. That is what keeps
  `test_influence_gateaux_drtmle.py`'s `1e-12` window, and every `ipsi` fit, a surface by
  construction rather than by measurement.
- **`psi` moved where the convention changed which mechanism later steps read**, by `0.003·se` on
  an ordinary clipping draw and `0.69·se` on a weak-overlap fit at a forced bound. The validation
  plan predeclared that as something to investigate rather than to reject a candidate for, and the
  investigation is the paragraph above: a different targeted state, not a different answer to the
  same one.

**1. The curve is labelled by its evidence now, not by its provenance.** The charge was that
`D = D* − D*_Q − D*_g` was *read off `drtmle`'s implementation rather than derived*, and that
every document here said so in a way that left a reader no way to tell what would change that.
Two things closed it and they are different in kind. The **provenance** is unchanged and stays
written down: the formula was read off that package's source, and
[§9](drtmle.md#refused-by-name)
records what else was, including the `gr1`/`gr2` naming inversion. What is new is the
**evidence**, and there are two independent pieces of it: Theorem 1's appendices, at a nonzero
`Q_r` (item 21), and the Gateaux derivative of the parameter on a law where one nuisance is wrong
on purpose (A1a). `docs/methodology.md`, `docs/user-guide.md`, `estimators/drtmle.py` and
`reduced_corrections`' own docstring now say which is which, in those words.

The distinction is the whole item and it is easy to collapse back. "Transcribed rather than
derived" is a fact about *where the code came from* and can never stop being true; it is not a
statement about whether the code is right, and leaving it as the headline invited a reader to
treat the two as one. What replaced it is not a softening — the refusal to compare against that
package's **numbers** is worded exactly as it was, and is [item
2](#closed-since-this-list-opened)'s standing decision rather than a hedge.

**2. `drtmle` parity, component by component — withdrawn, not done.** This is the one entry here
that closed on a **decision** rather than on evidence, and it is written at length because a closed
item is a paragraph and a withdrawn one is a standing refusal that a future reader will otherwise
re-propose. **The parity *piece* stays withdrawn**: no run of another implementation is a component
checklist here, no agreement with one is a release criterion, and no fixture exported from one is
admitted as a truth — not for `drtmle`, not for `tmle`, `tmle3` or `ctmle`.

> **One narrower thing has since been authorized and it is not this item.** A bounded differential
> **trace** — same frozen fixture, same initial nuisances, compare the trajectories to find where
> they first diverge — runs as F3, isolated
> from the package and from every test tier. It is an instrument for **localization** and never
> evidence, which is [stop-ship 17](#stop-ship) and
> the reversal. The reasoning below
> is what it does *not* overturn: everything here about agreement being evidence about a
> transcription stands exactly as written, and is why F3 may not justify a line of `src/`.

`CLAUDE.md` carries the rule and carries the narrowing with it; this is the reasoning.

Parity was never going to demonstrate what this package needs demonstrated. The page had already
said so twice without acting on it: *both packages descend from one source, so agreement is evidence
about the **transcription** and is blind to exactly the class of error item 21 is in.* Item 21 is
the worked example and it is not hypothetical — the mechanism correction's sign, read off a display
that both implementations read off, adjudicated in the end by the paper's appendices and by exact
arithmetic this repository already had. A parity run would have recorded R's sign as correct. It
would have been right, **by luck rather than by method**, and it would have made that reading
permanent.

The corollary is what cost the most to learn and it is why the replacement is not simply "an exact
law instead of R": **the exact-law instrument is blind in the same place.** At correct nuisances
`Q_r` and `g_{r,2}` vanish row by row, so every `test_influence_gateaux*` module passes against a
flipped sign too. Two checks that cannot fail against the same class of error are one check,
however different their machinery
([lesson 9](#what-the-sizings-got-wrong)). What is not blind is a
comparison against the theorem's own terms at a **nonzero `Q_r`**, which is
`tests/unit/test_theorem_drtmle.py`.

And the empirical record points the same way. Item 20 sat in this piece for two revisions on the
reasoning that a divergence between two arrays is what a component-by-component comparison locates.
The reasoning was sound and the premise was false — there was no divergence between two arrays —
and what found it was thirty lines, one fit and **no R**: recomputing the recorded score from the
returned state in the same process. That is
[lesson 8](#what-the-sizings-got-wrong), and it is the second time this
page filed a finding behind a fixture that would not have produced it.

**What survives is the decomposition, and it moves to A1a.** "Compare components, not `psi` and
`se`" was always the right instruction — several differences cancel at `psi`, and `psi` is precisely
what all three empirical means being zero makes insensitive to the corrections. Only the thing each
component was compared *against* was wrong. Each is now checked against its derivation, and the
[concordance's object table](drtmle.md#2-the-theorem-backed-contract) gains an
`evidence` column naming, per object, the test that pins it; the rows still reading `TODO` there are
what is left of this item, restated as tests to write rather than as a fixture to import.

The first review's recommendation A3 — *"cross-language validation at the
component level"*, called a release blocker there — is **declined**, with this as the written
reason. Its A4, the independent algebraic check, is accepted and is A1a's; that review's own argument
for A4 is the argument against A3, since it says in its own words that a cross-language comparison
can reproduce the same bug.

Two things this does **not** mean. Reading R's source, and recording in prose what was read off it,
stays — the curve's provenance, the `gr1`/`gr2` naming inversion, `fluctuateG`'s post-fit clip, the
`_NEGLIGIBLE / n` bar's R-shaped origin. Those are facts about how this code came to exist and the
concordance is where they live. And it does not lower the bar: the claim *that Python implements the
same algorithm as R* simply stops being one this repository makes. Two claims remain, and they are
the two that matter — that the algorithm satisfies the theorem, and that it helps in finite samples.

**23. A single-guard fit no longer subtracts a correction whose equation it never solved.** Found
by B1a's instrument on its first run against a `guard=("g",)` fit, which is what an instrument is
for. `reduced_corrections` did not branch on `guard` and neither did its caller, so a fit guarding
one nuisance subtracted **both** `D*_g` and `D*_Q` while solving one of the two extra equations —
and the unsolved one's mean was whatever it happened to be. Measured on `nonlinear_dgp` with `glm`
on both nuisances and **zero clipped rows**, so it was never item 20 wearing a different hat:
`2.8e-03` at arm 1 on the `n=400` draw it was found on, and `1.2e-03` and `3.1e-04` against a
`5.4e-06` bar — 225 and 58 times over — on the 600-row draw `tests/unit/test_drtmle_fit.py` fits
everything else on, which is now the regression fixture. Each arm's reported curve was off by
exactly its own number, and the two draws agreeing is what says this is the estimator and not a
seed. The
default `guard=("Q", "g")` was unaffected, which is why nothing here saw it: **no test in this
repository fitted a partial guard end to end**, and now one does.

The fix is where the guard already was. `ReductionFluctuation.guard` was on the record, serialised,
and read by `correction_check` one call *after* the curve declined to read it — so the change is
that `CorrectionParts` carries the guard and `total()` selects on it, `estimators/tmle.py`'s
`correction_parts` threads `reduction.guard` through, and the two now select from one place rather
than consulting the record twice. `guard` became a **required** keyword on `reduced_corrections`
and `reduced_correction_parts` with no default, which is the point: a default of both would make
the caller's mistake the fallback for the next one. `guard=()` raises there rather than returning
zeros, since such a fit fits no reductions at all and must not reach the corrections.

Three things fall out and each is worth having written down:

- **A fit's report got wider, not narrower.** The unsolved equation's correction is still
  recomputed and printed — as a new `diagnostic` row kind, held to no threshold, which is what
  stops a correct single-guard fit failing a check for a term nothing subtracts
  (`CorrectionCheck.correction_failures` gained one `row.solved`). Dropping the row would make a
  partial-guard report quietly smaller than a full one, which is the failure item 16 was about. It
  is also the number that *found* this.
- **The verdict is derived rather than written out.** `score_check` hard-coded "the curve reported
  is `D = D* - D*_Q - D*_g`", which is false under a single guard; it now composes the string from
  which corrections are in the curve, so it cannot drift from what `total()` did. Such a fit reads
  `D = D* - D*_Q`.
- **No `psi` moved and nothing on the default path moved.** `counterfactual_means` computes the
  estimate before the corrections enter, so this touches the curve, `se`, the interval, the bands
  and `estimate.score` — on partial-guard fits alone. And **no format bump**: `guard` was already
  serialised, so a reloaded fit selects what its estimator did. A pre-fix v10 file with a partial
  guard would carry a stored curve built with both terms, and none exists, because no such fit was
  ever run here — which is this item's own finding.

Pinned in three tiers, each mutation watched to fail before the test was kept. At the arrays
(`test_influence_drtmle.py`), against `test_remainder_drtmle.py`'s longhand terms at the
*wrong-on-purpose* nuisances — checked at the exact law too, where all three guards agree and
every array is exactly zero, so the fixture is load-bearing and lesson 2 is answered rather than
assumed. At the production call (`correction_parts` on a real partial-guard alternation), which is the tier
that fails when the guard stops travelling and the array tier does not. And end to end, on a
`guard=("g",)` fit of this module's own draw — 1.8s, because a `"g"` fit refits no reductions at
all. The mutations: revert `total()` to the sum (the curve equality, the centring, the report and
the verdict all go red — before the fix this was a *failing* fit), cross the guard semantics, have
`correction_parts` pass a literal `("Q", "g")`, drop the `row.solved` filter, and restore the
hard-coded verdict string.

The derivation was already in this repository, which is what made this a defect rather than a
question: `tests/unit/test_remainder_drtmle.py::_expansion` adds `d_g` **only** when `"Q"` is
guarded and `d_q` only when `"g"` is, and [the module's own
finding](#limitations-recorded-rather-than-fixed) — that one guard removes the whole first-order
remainder and two over-correct — is stated in exactly those terms. It was independent of A1a and
B1b and did not queue behind them.

**7. The relative-score exit criterion was a poor instrument — replaced.** The loop exited on
`|score| / mean|h|` against `spec.tol = 1e-10`, and `mean|h|` is `1e-3` to `1e-2` for equation
(10)'s covariate, so an absolutely negligible score read as a large relative one: on **68 of 96**
fits equation (10)'s relative score was above the tolerance while the worst absolute score was
under `1e-3` of `se/√n`. `targeting._solved` now accepts an equation on *either* ruler — the
relative test as before, or an absolute score under `_NEGLIGIBLE / n`, which is the bar
`score_check` already applies to the fit that gets reported. Asymptotic linearity asks for
`P_n D = o(n^-1/2)`; machine zero was never the requirement. It applies to **all three** equations
rather than to equation (10) alone, which was measured rather than assumed: on a 400-row `linear`
fit the round the loop gave up at had equation (10) at `2.3e-8` *and* equation (9) at `3.9e-8`,
with the joint likelihood flat to six decimals — the two trade off, so relaxing either alone stops
nothing. Equation (8), whose `1/g` is bounded below by the truncation, still stops on the relative
test, so a well-conditioned fit is unaffected. Refitting three processes at two seeds under both
rules, every fit moved from `stall` to `tolerance` and took a third to a tenth of the rounds —
`linear` 30 → 3, `nonlinear` 22 → 8, `weak-overlap` 36 → 11 — while the worst score `score_check`
sees was no worse and usually better, and `ate` moved by at most `4.1e-5`, which is `2.4e-4` of a
standard error. What loosened is the *loop's* internal stopping rule and nothing a reader is
shown: `score_check` still holds the reported fit to `1e-3·se/√n`, which is why it still fails 23
of 24 `weak-overlap` fits.

**14. Validity is not efficiency, and now the output says so.** The prose never called the
corrected curve efficient — checked, and the review's charge came back narrower than stated. The
*output* did: `score_check` signed a `DRTMLE` fit off with "the targeting step solved the
estimated **efficient** score equation" over three rows, two of which are the corrections, and
`inference/influence.py`'s first line called every curve in the module efficient. A corrected
fit's verdict now names what it solved and says the curve is `D = D* − D*_Q − D*_g`, valid under
weaker conditions rather than efficient under them; the paragraph is in the appendix, the guide,
`estimators/drtmle.py` and a paragraph of `reduced_corrections`, and the README cell says it in a
clause. A *plain* fit's verdict is byte-for-byte what it was — `README.md`'s transcript quotes it
— which `tests/unit/test_drtmle_fit.py` pins in both directions, watched to fail against deleting
the branch and against never setting the flag.

**16. A failing score check is now on the face of the report.** `score_check` was opt-in and
`estimators/drtmle.py` told the reader to run it "on every fit rather than assuming" —
documentation standing in for reporting, on an estimator whose only product is inference.
`TMLEResult.score_verdict` carries the verdict, derived from the fluctuations rather than stored,
and `summary()` ends with it whenever the check fails. **On every fit, not only a doubly-robust
one**, which is wider than this item proposed: the argument does not depend on which estimator
left the score unsolved. And **only** when it fails, which is narrower: a passing fit prints
nothing new, so every transcript in the README and the guide is unchanged and the line is worth
reading when it does appear. The interval is still printed — saying it is unlicensed is this item,
predeclaring which regimes are refused outright is
piece B2's and needs its evidence. Serialising
the records the check reads was a prerequisite and is format version 10.

**17. The weights claim is the narrow one now, and it is checked.** The docstring said `weights=`
"needs nothing said about it: the reduced regressions are fitted by weighted loss and every score
equation here is weighted". Both halves are true of the code and neither is the claim that needed
making — the derivation was read at an unweighted law. It now says what transports (the reductions
are `P_w`-conditional expectations because they are fitted by weighted loss, and the mechanism
they condition on and divide by is the `P_w` one because `nuisance.propensity` *is* the weighted
fit) and where it stops (an estimated weight). `tests/unit/test_remainder_drtmle.py` takes the
whole expansion at two tilted laws and keeps the wrong transport as a test: reductions at the
sampling law leave a first-order remainder a single guard no longer removes. Running that under
*both* weight functions is what found the blind spot now written into the module — a weight
reading `W` alone is a covariate shift, so it leaves every conditional alone and the mutation is a
no-op there. A **fitted** weighted run is still open and is an applied stress test, not this.

**18. `repeats=` averages what it says it averages, and checking it found item 20.** Each draw
runs its own alternation against its own refitted reductions — pinned by the two draws' `Qr`
differing and by `score_check` reporting three solved equations per draw — the report is the mean
of the draws, and no draw is dropped. The mutation this list proposed, "drop a repeat and watch
the averaged curve decentre", is **inert**: a centred curve carries its own `−ψ_r`, so the mean of
any subset of centred curves is centred. What a dropped draw moves is `psi` and the row count, and
the tests bite on those. What it did surface is item 20, which is a defect in the *fit* rather
than in this keyword.

**12. That change is now pinned by a test.** It was not, for a while, and the gap is the one
`CLAUDE.md` names: the whole 61-test `drtmle` suite passed identically before and after, because
every assertion in it is about the *reported* fit and the closing pass makes that fit the same
either way. `TestAnEquationStopsOnEitherRuler` in `tests/unit/test_drtmle_fit.py` unit-tests
`_solved` directly, and the absolute branch was deleted and the suite watched to fail — two of the
four assertions go red — before the test was kept. Asserting `exit_reason == "tolerance"` on a
fitted result was rejected: which exit fires is a property of the draw.

**The item's two loose ends closed with B2b, and the honest report of one of them is that it was a
restatement.** *The exit distribution under the current rule was uncharacterised* — it is now, and
it inverted: 87 of 96 fits reach the tolerance where 2 did. And *the loop's absolute bar was a
proxy for the one it cites*: `_NEGLIGIBLE / n` was justified as `score_check`'s
`DEFAULT_TOLERANCE · se/√n` with `se = O(n^(−1/2))` on the scaled outcome substituted in, which is
circular, since the loop runs before the estimate exists. It is now `_negligible_bar(n)`, stated as
a numerical criterion in its own right — a deterministic `c_n/√n` with `c_n = 1e-3/√n`, which is
the finite-sample rendering of the `o` in `P_n D = o_p(n^(−1/2))`. **The arithmetic is unchanged
and no fit takes a different exit**; what changed is that the criterion now rests on a property a
test can check rather than on an assumption about a quantity it precedes.
`test_the_bar_renders_an_o_and_not_an_O` asserts `bar(n)·√n` decreases and crosses a fixed level,
and the mutation is a bar of `1e-3/√n` — a legitimate-looking sequence rendering an `O` — under
which that product is flat and two of the class's five tests go red. The other half of the
separation was already in place: the standardised score `|P_n S_j|/sd̂(S_j)` is reported *beside*
the stopping rule by the sweep's *What the reported curve rests on*, not folded into it, and
whether a fit is entitled to a Wald interval stays `score_check`'s question at the realised `se`.

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

Thirteen lessons from the `DRTMLE` validation programme, kept when the rest of its record went
to the `drtmle-validation-archive-2026-08` tag. They are kept because the only thing a
retrospective is for is the next sizing, and because every one of them is about how to size and
check *any* piece of work here rather than about that variant. In one line each:

1. a refusal's stated reason is wrong about half the time — re-derive it before sizing the work;
2. the exact-law instrument goes blind wherever a quantity vanishes at the truth — write the
   mutation before the test, and watch it fail;
3. a threshold changed after seeing a failure needs the failure characterised first;
4. a test written after a change and never watched to fail pins nothing;
5. the sizings got the *size* about right and the **shape** wrong twice;
6. the claims that last longest are the ones no test can fail — the cheapest instrument for a
   prose claim is a reader with the source open, and the third review is that lesson arriving with
   the source itself attached;
7. a test can pin the wrong half of what it is named for, and **where a finding is filed is part
   of what it says**;
8. two numbers that should be equal and are not is not yet evidence of two states — recompute the
   recorded number from the returned state before looking for a second one;
9. a finding located in the code is not a finding adjudicated against the theorem, and parity with
   a reference implementation is blind in exactly the same place — which is why the parity piece
   is retired rather than merely deprioritised;
10. a display is not a derivation — when a source and an implementation disagree, check whether the
    source disagrees with *itself* before changing the code, because item 21 did;
11. before building an oracle, check whether the quantity collapses onto one already here **at the
    value the check has to be taken at** — A1a's was sized as a whole analytic DRTMLE functional
    and turned out to be a comparison against an EIF written years earlier, because the union
    model is where the theorem applies and the corrections vanish into `1/g_0` there;
12. the closing pass is an **anaesthetic** — when a stage downstream of a loop recomputes what the
    loop was supposed to establish, no test of the output can test the loop, which is why item 12
    and B2a's stale-score restatement were both invisible to whole suites. Remove the asymmetry,
    then pin the invariant one level down; a structural pin is the third choice and reads like the
    first two;
13. a stop-ship's stated **reason** can carry the error the stop-ship was written to catch, and
    only building the thing it talks about will find that. A conclusion protects the case it names;
    a reason is what a reader generalises from, so a wrong one points the guard-rail at the wrong
    hazard while looking right. When a document says an instrument is blind, check *why* by
    building what it says the instrument cannot see — stop-ship 14 survived two revisions because
    the construction it was about did not exist yet.

## On native acceleration

**The instrument had stopped running, and this section was folklore until it was fixed.**
`benchmarks/bench_tmle.py:102` built an `InitialFit` positionally, as `(observed, at_one, at_zero)`;
that dataclass has taken two fields — `observed` and an `arms` dict — since counterfactual
quantities were keyed by arm. `bench_targeting` is the first section `main()` evaluates, so
`python benchmarks/bench_tmle.py` and `nox -s bench` both died on a `TypeError` before timing
anything. Every number below the profile table dates from before that, and could not be
reproduced by anyone who tried. It runs again, and the re-measurement is at the end.

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
nuisance fit, so they stay scikit-learn-bound.

### The re-measurement, and what numba answers

`bench_tmle.py` now covers the kernels that grew since it was written, and the longitudinal
prediction above is a measurement rather than a prediction. Shares of one full fit, n=2,000,
four cores:

| | `library="glm"` | `library="default"` |
| --- | --- | --- |
| full fit | 0.354 s | 8.28 s |
| targeting (Newton) | 0.31% | 0.01% |
| the whole dataframe boundary, every backend in and out | 1.51% | 0.06% |
| an `LTMLE` fit's four node fluctuations, as a share of that fit | 0.22% | 0.01% |

Three conclusions, and the middle one answers a question that used to be settled by assertion.

- **numba buys nothing, at any size.** The targeting Newton is the most favourable kernel in the
  package for a compiler — neither BLAS-bound nor scikit-learn-bound, just `(n, K)` passes and
  per-trial temporaries — and a hand-fused `njit` version of its loop and line search agrees with
  numpy to `2.8e-17` and runs **1.2× faster at n=2,000**. That speed-up is per-*call* overhead
  and does not grow: measured directly, the ratio is 1.07× at n=2,000, 0.67× at n=20,000, 1.02×
  at n=200,000 and 1.05× at n=2,000,000 — a wash, on top of a 4-second one-off compile. Which is
  the expected answer once stated: the numpy loop is already `x @ eps`, `x.T @ (…)` and a
  vectorised `exp`, so its inner work *is* compiled BLAS and SIMD, and a scalar loop has nothing
  left to remove. A compiler pays where the interpreter is in the inner loop, and here it is not.
  The arm lives in the benchmark (`--skip numba` to leave it out, `pip install -e '.[bench]'` to
  include it) so the next person to ask gets a number instead of an argument. Nothing under
  `src/` imports numba.
- **The dataframe boundary is not a bottleneck, so the internals stay numpy.** Ingestion and
  emission across pandas, polars and arrow-backed pandas together are 1.5% of the cheapest fit
  and 0.06% of a realistic one — and, per the next section, **1.5% and 0.04% asymptotically**,
  so this is not a small-*n* artefact. There is no share here for a columnar engine to win, and
  scikit-learn takes contiguous numpy regardless: the internals are numpy because that is what
  the arithmetic and the estimators both want, not by omission. Worth stating in those terms
  because "use polars internally, it is faster" is a reasonable thing to assume and wrong for
  this shape of work — polars wins at joins, group-bys and IO, none of which is on this path.
- **Longitudinal TMLE is scikit-learn-bound, as predicted.** Its node fluctuations are 0.22% of
  a `glm` fit. The loop over timepoints is Python and each body is a nuisance fit.

### At several million rows

A share measured at n=2,000 does not transfer, and not for the reason one might guess. A full fit
carries a few hundred milliseconds of fold setup that a kernel timed on its own does not, so at
small *n* **every kernel looks cheaper than it asymptotically is**. `bench_tmle.py` therefore fits
`seconds ≈ fixed + per_row · n` per kernel and reports `per_row` as a share of the fit's — the
limit each share tends to. (Fitting a plain power law instead is a trap worth naming: it charges
the per-call overhead to the exponent, which then reads `n^0.18` for a full `glm` fit and, worse,
inverts the numba comparison outright.) Nothing is run at these sizes; `--project` sets them.

Per-row costs, fitted over n = 2,000…100,000, as a share of one fit's per-row cost:

| kernel | share of a `glm` fit | share of a `default` fit |
| --- | --- | --- |
| `solve_one_step` (the universal least-favourable walk) | **82%** | 2.2% |
| multiplier bootstrap, `kind="rademacher"` | 17% | 0.46% |
| `solve_projection`, identity link | 17% | 0.44% |
| gram einsum at `optimize=False` | 9.1% | 0.24% |
| targeting (Newton) | 7.4% | 0.20% |
| the whole dataframe boundary | 1.5% | 0.04% |
| multiplier bootstrap, `kind="normal"` | 0.69% | 0.02% |

A `default` fit costs **37×** more per row than a `glm` one, which is the whole of the difference
between those two columns. So the verdict holds where it matters: with a real learner library
nothing cleverly-authored reaches 3% of a fit even at five million rows.

**The one scenario that inverts, and it is not fixed with a compiler.** `library="glm"` at
millions of rows is a legitimate production choice, and there `solve_one_step` asymptotically
*dominates the fit* — 82%, against 7.4% for the Newton solver that answers the same question. The
gap is algorithmic: the one-step walk takes up to 20,000 Python iterations, each doing a full
multi-arm `apply_logistic` and `score_columns`. Anyone hitting that should reach for
`targeting="newton"` first, which is 11× cheaper per row and already the default. Likewise
`multiplier_kind="normal"` is 25× cheaper per row than the resampling default and its closed form
never allocates — though `"rademacher"` stays the default for the reason above: `"normal"` cannot
see the leverage a `1/g(W)` covariate produces under weak overlap.

**And time is not what breaks first.** Two allocations grow faster than `n`: the multiplier
bootstrap's `(256, n)` chunk and the conditional-density learner's `≈ n·bins/2` long design. At
n=5,000,000 each is around **9.5 GB**. That is the binding constraint on this library at scale, it
arrives well before any arithmetic does, and no amount of native code addresses it — `kind="normal"`
avoids the first by never forming the array at all.

**The first of those two is now fixed, and the fix says the profile above pointed at the wrong
thing.** "92–95% multiplier generation" reads as an argument about the random draw; split at
n=100,000 it is 3.5 ms drawing the packed bits, 1.7 ms unpacking them, 12.6 ms in the `dgemm` —
and **159 ms, 89%, expanding one bit per element into a 205 MB float64 array**. Expanding in
place into a buffer sized by a byte budget rather than by a replicate count makes the path
3.4–3.9× faster and puts its buffer on a **32 MB budget with a four-replicate floor** — so it
is 32 MB up to `n = 2²⁰` and `32n` bytes above that, 160 MB at five million rather than 9.5 GB,
and it is a buffer figure rather than a working set. With the seeded stream
untouched and the critical value bit-identical. So the allocation that broke first at scale is
gone without a compiler, and the remaining one — the density learner's long design — is now the
binding constraint on its own. `docs/benchmarks/bootstrap_numpy.md` has the measurement.

One change came out of it. `np.einsum` defaults to `optimize=False`, which for three or more
operands means numpy's own nested-loop kernel rather than a pairwise contraction through BLAS —
so the four-operand Jacobian term in the MSM projection, `"ijp,ijq,ij,i->pq"`, was **14× slower
than the same arithmetic reshaped into one `dgemm`**. `_projection_state` now passes
`optimize=True`; it runs once per Newton step *and* once per line-search trial under a
non-identity link. The identity-link closed form in `solve_projection` deliberately keeps the
unoptimised spelling: reassociating moves the last bits, which turned
`test_the_identity_link_is_the_closed_form_bit_for_bit` red, and the whole projection is around
1% of a fit — a fraction of a fraction, against a regression pin. Do not "finish the job" there.

**HAL remains the trigger**, unchanged and untouched by any of this.

The measurements are reproducible — rerun the benchmark before revisiting this, with
`--library default`, since `glm` is the cheapest preset available and inflates every other line's
share several-fold.

### The wider instrument, and what it changed

Everything above rests on **one** compiled kernel — a hand-fused Newton loop — and that kernel
is the *least* favourable case a compiler could plausibly be offered: its inner work is already
`x @ eps`, `x.T @ (…)` and a vectorised `exp`, so a scalar loop has nothing left to remove.
Reading "numba buys nothing, at any size" off it is reasoning from a negative control, and
`benchmarks/numba/` is the instrument that stops doing that. Same question, every post-nuisance
kernel in the package, core count as an explicit parameter, a correctness gate on every row.

The conclusion above **holds where it was measured and does not generalise**, which is the
correction. Fused row-wise kernels, indexed accumulation and the compiled recursions are 2–12×,
and the multiplier bootstrap's `(chunk, n)` array — named above as one of the two allocations
that break first at scale — need not exist at all.

**And then acting on that produced a second correction, in the other direction.** Two of those
ratios were against the *shipped* numpy path rather than against a competent one, which is not
the same baseline: the bootstrap's cost was the float64 expansion and not the draw, and
`cluster_sums`'s was an `np.unique` re-deriving an encoding `encode_clusters` had already
produced. Written properly in numpy, the bootstrap is 3.4–3.9× and the compiled kernel's
advantage over cluster aggregation falls to 1.02× at five estimands and **0.74× at a million
rows**. A third — the LTMLE mask fix — is real and `O(T²n)` → `O(Tn)`, and is 0.06% of a fit,
because the ratio quoted for it was of a cached-nuisance region that excludes the learners by
construction. [`production_plan.md`](benchmarks/production_plan.md) is the adjudication and
[`findings.md`](benchmarks/findings.md) carries all four corrections in its body.
**`numba` is still a benchmark-only dependency, and the case for changing that is weaker than
the measurement first suggested.** Read
[`candidate_inventory.md`](benchmarks/candidate_inventory.md) first: it is the profile the rest
was sized against, and three of the things it is natural to expect turn out to be false. In
particular the largest package-owned cost in a DR-TMLE `retarget` and in an LTMLE fit is
**`threadpoolctl`**, entered once per learner fit at 1.4 ms a time — 57% and 40% respectively —
which is not a compilation question and is not fixed by one.

**And then a third correction, which does not move the verdict but narrows what it covers.** An
external review of that work found four defects in these documents, all of them confirmed
against the source. Two are arithmetic claims that were simply wrong — the bootstrap's buffer is
a 32 MB *target with a four-replicate floor*, not a constant, and the ~20-estimand crossover for
`cluster_sums` does not follow from its own table, which reads 4.31× at seven. The other two are
about the instrument: every recorded number was taken in randomised **block** order rather than
the interleaving three documents claimed, and **no CI job has ever run above two cores** — the
`full` tier passed `--num-cores 1 2` alongside a config sweeping to eight, and the flag replaces
rather than narrows. So the defensible statement is that the *measured* serial and low-core
workloads do not justify a runtime dependency; it is not evidence that a compiled kernel would
fail to help a large repeated workload on 8–32 physical cores. That question is open and the
`runner:` dispatch input is now how it would be answered.

```bash
pip install -e '.[bench]'
nox -s bench-numba              # the kernels
nox -s bench-numba-pipelines    # the denominator: post-nuisance share, per flavour
```
