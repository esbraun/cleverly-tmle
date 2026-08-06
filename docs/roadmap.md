# Roadmap

What has landed, what is open, and where native acceleration does and does not pay.

**One thing is open**: `DRTMLE`, the doubly-robust-inference variant, which is written and
tested and not finished. [What is still open](#what-is-still-open) is the list, grouped into
pieces of work, each of which is a pull request rather than an errand.

**No known defect is live.** The last one — the reported curve was not centred wherever the
mechanism truncation bound — closed with
[B1b](#b1b--the-theorem-conforming-targeting-decision), and what is left is **a demonstration and a
widening of scope**. That is a different kind of list from the one this page opened with, and it
does not lower the bar: *done* still means a demonstration that the interval attains nominal
coverage where a plain `TMLE`'s does not.

**The study has now run, and its answer is that the demonstration is made and the estimator does
not clear.** [C3c](#what-c3c-measured) dispatched 6,000 fits over two seed batches, entered the
regime it committed to for the first time in three attempts, and produced the gap the variant
exists for: `TMLE` at `0.532` and `0.472` against `DRTMLE` at `0.844` and `0.848` in `q-drift` at
`n = 2,400`, a paired `+0.312 ± 0.031` and `+0.376 ± 0.033`. **And `DRTMLE` does not attain nominal
coverage anywhere in it.** So gate 1 fails at four clauses and gate 2 passes its shortfall clause
and fails its invalid-rate one. Both halves of that are results;
[the gate readout](drtmle/coverage-study.md#the-gates-read-out-clause-by-clause) is the sentence
each clause earned.

**What the study did *not* establish is why**, and [piece E](#e-what-c3c-handed-back) is that
question. `√n R_remaining` — item 13, and Theorem 1's own assumption — is flat over `600`–`2,400`
**with the three reduced regressions fitted by `glm`**, a configuration whose own consistency
[the concordance marks `unverified`](drtmle/theorem-concordance.md). So the measurement says *this
estimator, at this reduction, does not meet the condition at these sizes*; it does not say the
condition is unmeetable, and item 13 stays open on exactly that distinction.

**The newest thing on this page is [E2R's instrument](#what-e2rs-instrument-landed), which is the
reboot's WP1 built and not yet run.** The rung the three reduced regressions are fitted at is now
**selected** against a measured ranking rather than shipped — which is what E2's own falsifier asked
for — on four blocks rather than three, so the block that certifies a rung is not the block that
chose it; gate B reads **five** metrics where it read three, the two new ones being the composites
`q_r/g` and `g_{r,2}/g_{r,1}` the correction actually divides by; and the negative control is coarse
enough to be rejected on all five. **Four of those choices came out of three sandbox pilots
differently from how they were written**, which is the return on sizing an instrument before
dispatching it: `bins(4)` is not coarse enough to be a control at a bound-active divisor; a ranking
taken at the initial pair ranks `h3` at a divisor no fit uses; the block-size lever's own falsifier
was **run and did not fire**, so the reference is fitted on twice E2's points and the audit's one
failing clause resolves; and — the one worth carrying away — **a selection judged on point estimates
cannot be certified by a clause read on intervals**, which the first selection rule was and which
its own gate rejected. **No verdict here is E2R's** — the decision run is owed and dispatches from
`main`, exactly as E2's did, and no pilot's comparison column is read.

**The reboot itself is the revision before that, [a rebooted piece
E](#the-reboot-and-the-three-work-packages), and it came from an external critique of the state
after E2.** It reads the page against Theorem 1
and against the study's own artefacts, and it lands four things. **Two are corrections this page
has checked and accepted**, and both are about how the evidence was described rather than about a
number:

- **The `se` shortfall is a *symptom*, and every revision since C3c has called it a named defect.**
  The sentence — `σ²ₙ` "treats the reduced regressions as known, so their estimation error is in
  `ψ̂`'s spread and not in the variance estimate" — reads as though the variance estimator carried a
  repair of its own. It does not. `σ²ₙ = Pₙ{D* − D_A − D_Y}²` **is** Theorem 1's variance, the
  [concordance states it in the theorem itself](drtmle/theorem-concordance.md) and
  `test_theorem_drtmle.py::TestTheReportedVarianceIsTheorem1s` pins the reported interval to it. What
  is unestablished is the theorem's *premise* — which is the premise item 13 already fails — so the
  `se` ratio and the remainder are **one** open question, not two. **The study's own column is what
  settles it rather than the argument**: an omitted non-negative term predicts an `se ratio` below
  one in *every* cell, and `g-drift`'s reads `1.157` and `1.156`.
  [The withdrawal is here](#the-se-shortfall-is-a-symptom-and-what-withdrawing-it-costs).
- **The split of `q-drift`'s eleven points into "about five `se`, six bias" is an artefact of the
  order the two counterfactuals were applied in.** The `0.898` is reproducible —
  `Φ(1.96r − b) − Φ(−1.96r − b)` at `r = 0.903` and `b = 0.664` gives `0.8982` to four places — and
  the *same* arithmetic run the other way gives `0.9232` for removing the bias alone. `+0.040` then
  `+0.052`, against `+0.065` then `+0.027`: they do not add, the split is not identified, and under
  either ordering the bias is the larger term.

**The third this page has checked and narrowed**, because it repeats a sentence of this page's own
that the table under it does not carry: *the repair E2 names is a rung per **cell**, not a rung per
reduced regression*. At the gate's own reading — a bootstrap interval over draws — no cell has one
reduction favouring `spline(8)` while another favours `spline(16)`. What flips is the **cell**:
`spline(8)` beats the reference in `q-drift` at both sizes and *loses* to it in `g-drift` at `600`,
on `gr1`, `+3.132e-05 [+5.19e-06, +6.44e-05]`. The per-regression reading is a point-estimate
reading of a gate defined on intervals.
[The correction is here](#what-e2-measured-and-why-it-did-not-branch).

**A fourth thing it does is close a candidate nobody had written down**, and that one needed no
correction because [item 25's witness had already measured
it](#the-three-candidates-ranked-by-what-the-study-measured): a small `g_{r,1}`, or an active
truncation, is *not* what makes `q-drift` fail. `min gr1 margin` is `0.117` to `0.343` across all
twelve cell-runs and the initial mechanism clips essentially nowhere. The column exists because C1
was made its own pull request for it, which is the return on that decision.

**Underneath all four, [E2 ran and did not
branch](#what-e2-measured-and-why-it-did-not-branch), and the reason is the falsifier its own rule
wrote down in advance.** Three cells of four are `unresolved` **on gate B**, every one on the same
clause — a coarser rung is the better estimate of at least one reduced regression. The fourth cell,
`g-drift` at `n = 600`, passes its gates and reads **`moved`** — `-1.7893 [-2.2739, -1.3569]` against
a margin of `±0.8997`, in the direction candidate 1 predicts. **So candidate 1 is not dead — nothing
reads `equivalent` — and it is not established either**, since the cell C3c's flat column is most
sharply read in cannot be read at all. The unread differences are large, and
[the section says why that sentence stops there](#what-e2-measured-and-why-it-did-not-branch).

**The gate that turned out to have the most teeth is still the exact-law control**, and the dispatch
has now added a second: the provider reproduces the law's own reductions at the alternation's exit
state to `3e-14`, and the same law makes exact the reason the gates exist at all — a two-bin
reference is wrong about `Q_r` by more than half its own magnitude while its `ψ` sits inside a
twentieth of one standard error of an exact reference's. **A gate reading the estimate would have
passed it** — and, on this run, would have branched the whole piece on three cells whose reference
another resolution beats.

**The other newest thing is how much of that flatness was the instrument, and it is now
measured rather than inferred.** `P₀D̂` is integrated on a **randomised** quasi-Monte Carlo rule —
the corrected curve is affine in `Y` and binary in `A`, so two of its three coordinates close in
closed form and a Sobol quadrature in `W` is all that is left, with an independent scramble per
replicate. [E1b](#what-e1b-measures) then measures each rule's own error from **replication at a
fixed fit** rather than from refining a grid, which makes the attribution identified:
at `n = 2,400`, **the evaluation draw C3c used accounts for essentially all of the across-draw
variance of `√n R_remaining`** — `0.991 [0.94, 1.05]` in the Tier-2 cell the rate is read in, and
`1.003` and `1.008` at Tier 1 — against `0.64` to `0.82` at `n = 600`. The grid's own share at the
same rung is `0.000` to `0.081`.

**[E1's two statements of the same thing are withdrawn](#what-e1-landed-and-what-e1b-withdrew), and
the tier-1 headline was about right by luck.** It read `1 − s²_grid/s²_draw` as that share, which
identifies it only if both rules' errors are mean-zero given the fit — a *fixed* grid's is not — and
it bounded the residual with a nested ladder, which a successive difference cannot do. Two of its
four Tier-2 cells came out uninformative in consequence and none came with an interval. **No reading
of the rate is affected either way**, because none was ever taken here.

**The instruments that got there all landed first.**
[C1](#what-c1-landed) is the harness, Tier 1's prescribed nuisance sequences with their
drift coefficients committed and verified, and [item 25](#the-supported-contract-and-item-25)'s
per-fit truncation witness; [C2](#what-c2-landed) is **Tier 2** — both nuisances fitted, the good
one a smoother at a committed bandwidth sequence — and the **evaluation companion** that makes
`P₀D̂` computable, so item 13's columns exist for the first time; and [C3b](#what-c3b-repaired) is
the **second declared coefficient** — the estimator's bias rather than the plug-in remainder — a
Tier-1 injection built to hit it, and the three pre-flight conditions as a verdict table the
harness prints and the workflow says to read first. Tier 2 also came in cheaper than
the tier it replaces: 5.4s to 7.4s a fit against the 43s this page costed C from.

C1's first run moved a scope claim rather than a number — and that witness immediately found the contract's condition is not the
ordinary case this page had read it as: a sixth to a third of *well-overlapped* draws exit with the
targeted mechanism pressed against a bound, because equation (9)'s covariate vanishes where the
outcome regression is right. Item 4 with the nuisances swapped. The fits are sound; what moves is
which estimator their numbers are evidence about. [C3a's pilot](#what-c3as-pilot-measured-and-why-the-study-is-not-dispatched-behind-it)
has since read that share as a **rate** — `0` to `20%` over fifty draws a cell, falling with `n` —
so the "sixth to a third" above is the high end of one six-draw reading rather than the ordinary case.

**And C3a's pilot moved the design rather than a number.** It falsified the sizing it was run to
check: Tier 1's regime-entry column reads its declared coefficient to four decimals while the plain
interval does not under-cover at all, because that column is the **plug-in** remainder and a fit's
bias is the same expression at the **targeted** regression. No defect surfaced and no score or
identity check failed anywhere in 600 fits; what was wrong is the instrument's premise.

**[C3b](#what-c3b-repaired) has repaired it, and the newest thing on this page is what the repair
turned out to be.** The pilot left the mechanism as a hypothesis and proposed a projection; the
algebra, once written, says the estimator's bias is a **linear functional of the injected shape**
against a second computable weight — so `b` is a coefficient a design can be *built* to hit,
exactly as `c` always was, and the repair is a 2×2 solve rather than a re-orthogonalisation.
Tier 1's root-`n` bias now reads `+1.93 / +2.17 / +3.31` where the pilot read
`−0.22 / −0.56 / +0.11`. Two of the pilot's readings move with it: the ratio was **436** and not
twenty — the pilot could bound it, not measure it — and the design's opposite-arm signs, which make
`c_ATE` a sum of magnitudes, **do not survive targeting**, so `b_ATE` had been a difference. The
study waits behind the three pre-flight conditions, which are now a table rather than a paragraph.

**The cross-fitting construction has come off that list**, and how it came off is worth a line
because it is not what this page expected. [A1b](#a1b--the-cross-fitting-construction) held two
tracks open — a proof for the pooled construction *or* a nested reference estimator — and writing
the argument showed they are one track: the residual the proof cannot reach is `Δ_k`, which is by
definition the difference between the two constructions. The entropy half then closes on a
structural fact the module docstring had been stating all along, that **the reductions are
univariate**, so the Donsker condition cross-fitting exists to avoid is available for them whatever
the primary nuisances did. The stability half is measured, supported in all six cells, and **not
shown** — which is why the wording throughout says so.

**The scope decision is the newest of them and it came from reading this page against its own
concordance.** B1b left the score and the curve one expression at one state; it did not make
Theorem 1's expansion a statement about the constrained estimator, and the assumption matrix had
said so in two rows the whole time. So the guarantee is now claimed for a fit whose truncations are
inactive — which is what the ordinary cells measure, and *not* what the weak-overlap ones do — and
bound-active fits are reported as empirically supported and outside the theorem. That is [the
supported contract](#the-supported-contract-and-item-25) and item 25.

**The sweep has run**, four dispatches of it, and [what it
measured](drtmle/investigation-log.md#what-the-b2b-dispatch-measured) closed items 12 and 19, took
the weak-overlap product decision, and rewrote limitations 4 and 6 from their own numbers. The
headline is that the alternation's exit distribution **inverted** — 87 of 96 fits reach the
tolerance where 2 did — and that `weak-overlap`'s score check now fails on **0 of 24** where it
failed on 23, on draws whose overlap is unchanged. So what is left is a demonstration, a widening
of scope, and one clause of one frozen rule. **That same dispatch supplies the scope decision's evidence**, in columns it was not run
for: `clip share` is `0.000` on the three ordinary processes and `0.231` to `0.338` on
`weak-overlap`, whose `margin` is exactly zero at both sizes it is recorded at. Those are the two
regimes item 25 separates, already measured.

That grouping and its order are a revision, three times over. An [external
review](drtmle/review.md) of this page and the code behind it read the plan against Theorem 1 of
Benkeser et al. (2017) and found the definition of done right and the route to it short by two
conditions, which are now items 13 and 15. A second review turned that into a dependency-ordered
execution plan, and checking *its* central claim — that the returned state and the reported curve
are read off different arrays — **found the cause of item 20**, which is not that: it is one array
read under two truncation conventions, and it accounts for item 11 as well. A third review read
the result against the **2016 working-paper version of the theorem**, which nothing here had had
in hand before. It accepted that diagnosis, and it found two things the diagnosis could not have
found and neither could any test in this repository: the mechanism correction's **sign** appeared
to disagree with the theorem (**item 21**), and the algorithm's **update order** is not the
theorem's (**item 22**). It also showed that the fix as planned could not be executed as planned,
because [piece B1](#b-the-loops-exit-and-whether-what-it-leaves-is-what-gets-reported) both
preceded and depended on piece A — so B1 splits into **B1a**, an identity and safety
patch valid under every eventual convention, and **B1b**, the targeting decision that had to wait
for the theorem. It no longer does: [A1a](#a1a--the-theoretical-audit) answered what it was waiting
for, and the answer was that the theorem offers no convention to match.

**Then the working paper itself arrived** — UCB Biostatistics paper 356 — and reading it rather
than a transcription of it closed both of the third review's items — the **sign in favour of the
implementation**, on that paper's own appendices, which contradict the display the charge was
filed from; and the **update order**, because the paper states its own exit as the three score
equations and so prescribes a fixed point rather than a route. That is
[lesson 10](drtmle/investigation-log.md#what-the-sizings-got-wrong), and it is why this page no
longer opens with a stop-ship.

**Piece A has since split in two as well, and the reason is not B1's — worth keeping the two
apart.** B1 split because one half preceded the other and depended on it. A1 splits because item 15
— the cross-fitting construction — shares evidence with nothing else in the piece: everything else
there was a test to write or a reading to record, and that item is a proof to find or a second
estimator to build. So **A1a** is the theoretical audit and has landed, and **A1b** is item 15
alone; the half that unblocks B1b no longer waits on the half that unblocks nothing.
[Link 2 of the four](#what-is-still-open) — derivation-anchored correctness — closed with it,
except for item 13's rate, which [piece E](#e-what-c3c-handed-back) carries.

**This page is now one of five**, because it had become a status page, a methodology review, a
forensic report, an implementation specification, a test plan, a simulation protocol and a project
history at once, and that density made the dependency plan harder to execute than the plan itself
was:

| document | what is in it |
| --- | --- |
| **this page** | status, the pieces, what blocks what, the release gates, the definitions of done |
| [theorem concordance](drtmle/theorem-concordance.md) | the theorem objects, the assumptions and their grading, the appendix-B terms, the paper/Python/R mapping, the cross-fitting status |
| [validation plan](drtmle/validation-plan.md) | the fixtures, the candidate targeting algorithms, the benchmark columns, the coverage *specification*, the frozen decision rules, the mutations |
| [coverage study](drtmle/coverage-study.md) | the *design* the specification is realised by: the two cells, the committed drift coefficients, the re-timing, and what Tier 1 has measured |
| [investigation log](drtmle/investigation-log.md) | item 20's discovery, the clipped-row measurements, the hypothesis that was dropped, the convergence sweep, the runner history, the lessons |

The fourth row is the newest and the pair it makes with the third is deliberate: a **specification**
says what a study has to contain to be believed, and a **design** says what the cells are and what
constants were committed. Keeping them in one file is how a rule gets restated next to the numbers
it judges, and a rule restated is a rule that can differ from itself.

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
| **No R in this repository and none in CI**, and no parity test against another implementation | two checks that cannot fail against the same class of error are one check: Python and R descend from one source, so agreement is evidence about a transcription and not about a derivation | nothing. It is a retirement, not a deprioritisation — and referring to another implementation *in prose* was never what was refused. The narrower *differential-diagnostic* case for it is [answered below](#a-differential-diagnostic-against-r-considered-and-refused) rather than left unnoticed | [What the sizings got wrong](#what-the-sizings-got-wrong), lesson 9 |
| **A study that selects and then certifies runs on two disjoint cohorts of draws**, separated by a commit of the frozen selection | disjoint quadrature blocks split the *integration* noise, and the simulation draw is the independent unit: a rung chosen across a set of draws and certified on the same set is a data-dependent selection assessed on the sample that made it. The disjointness is checked on the **data seed**, since two draws sharing one under different splits are the same rows twice | a study whose selection is not data-dependent — a rung shipped rather than measured is one cohort's work, which is what E2 was | [§8's decision protocol](drtmle/validation-plan.md#the-decision-protocol-frozen-before-the-dispatch) |
| **A fidelity gate passes on a non-inferiority bound, never on failure to show superiority** | an interval containing zero establishes neither equality nor adequate approximation, so a gate read that way certifies whatever it cannot resolve — and E2R's own record called two rungs "genuinely indistinguishable" on exactly that reading | a margin that cannot be tied to a tolerable change in the reported column. The two composites' is `(δ/3)²/(n·weight_scale)` by Cauchy–Schwarz; the three componentwise ones have no such transfer and take a share of the negative control's measured distance instead | [§8's decision protocol](drtmle/validation-plan.md#2-the-fidelity-clause-is-non-inferiority-against-a-margin-declared-in-advance) |
| **Nuisance fits run single-threaded**, with one `ThreadpoolController` per process | parallelism belongs across folds and candidates rather than inside each small fit; building the controller per entry was 57% of a DR-TMLE `retarget` | a fit large enough that one model wants the machine — `set_thread_limit(None)` is the lever, not a code change | [`thread_limit_profile.md`](benchmarks/thread_limit_profile.md) |
| **`tracemalloc` is the memory instrument** | it *does* see numba's allocations, through all three CPython allocator domains — the caveat that said otherwise was wrong and was measured to be wrong | a question about resident memory rather than allocation, or a library calling `malloc` directly. That needs an incremental-RSS arm *beside* this column, not instead of it | [`production_plan.md`](benchmarks/production_plan.md) §1.3 |
| **Benchmark write-ups live in [`docs/benchmarks/`](benchmarks/)**; `benchmarks/results/` is generated output and is git-ignored | a `results.jsonl` from a four-core container reads as a fact about the package rather than about that box | nothing | [`docs/README.md`](README.md) |
| **A dispatched *study*'s per-replicate rows are manifested, and archived rather than left in CI.** The row above is about generated benchmark output and this is its exception, not a contradiction of it | a CI artefact has a 90-day retention, and C3c's four are the only copy of the 6,000 fits every gate verdict on this page is read from. A summary table is not the evidence, it is a transcription of it | nothing. The cost is a manifest per study and a few megabytes | [the manifest](drtmle/study-manifest.md) |

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
    concordance's §4](drtmle/theorem-concordance.md#4-the-sign-discrepancy-item-21--resolved);
    `tests/unit/test_theorem_drtmle.py` pins it, watched to fail against a flipped sign in the
    library itself;
  - **the reported curve is not centred whenever the targeted mechanism leaves the truncation
    bounds** — on roughly a quarter of ordinary splits and on 23 of 24 weak-overlap fits, at
    `2e-05` to `7e-04` where a solved fit sits near `1e-09`, while the loop's own three rows all
    report their scores solved to `1e-11`. That is items 20 and 11, and they are **one defect
    with a located cause**: equation (9) is solved against the *raw* tilted `g*` and the `D*_g`
    the curve subtracts reads the *truncated* one, so a single clipped row of 600 is enough to
    decentre the curve while every fluctuation row still reports `1e-11`. It is a `DRTMLE`-only
    defect and it is not confined to poor overlap. [The investigation
    log](drtmle/investigation-log.md#item-20-from-discovery-to-cause) carries the measurements.

  **[B1b](#b1b--the-theorem-conforming-targeting-decision) has landed and closed it.**
  `solve_bounded_mechanism` solves equation (9)'s score at the truncated tilt — the expression the
  curve carries — and the alternation carries that truncated array forward, so the two are one
  expression at one state. Measured on all four fixtures the defect was characterised on,
  including `weak_overlap` at a forced `g_bounds=(0.15, 0.85)` where 375 rows clipped: every
  identity at `1e-17` or better, every final score `1e-10` against a bar near `5e-06`, every score
  check passing. A fit whose bound never binds is bit for bit what it was.
  It was caught *by name* first, which is why the fix is checkable rather than asserted:
  [B1a](#b1a--the-identity-and-safety-patch) made `res.validation.correction_check()` recompute
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
  rest. Three of its items are there because the [first review](drtmle/review.md) put them there:
  the theorem's *other* assumption beyond the three score equations (item 13), the cross-fitting
  construction the reductions would need to satisfy it (item 15), and the claim that the
  alternation converges (19). Four more of that review's items — that the corrected curve is the
  efficient one (14), that a fit whose score check fails still reports an ordinary Wald interval
  (16), that weights need nothing said about them (17), and that `repeats=` averages what it is
  averaging (18) — were piece 0 and have [landed](#closed-since-this-list-opened)

## What is still open

**Done still means one thing: a demonstration that `DRTMLE`'s interval attains its nominal
coverage where a plain `TMLE`'s does not.** That is [piece C](#c-the-demonstration) below, and
nothing less clears the variant. What the reviews changed is not that bar but what it takes to
*believe* a demonstration that meets it. Coverage is one link of four, and each link can hold
while another fails:

**[C3c](#what-c3c-measured) has now run that demonstration and the bar is not met.** `DRTMLE`'s
interval is much better than `TMLE`'s where the design says it should be — `0.844` against `0.532`,
a paired `+0.312 ± 0.031`, reproduced — and it is **not nominal anywhere**, the best reading in the
study being `0.880`. The bar above says *attains*, so the variant stays unfinished. **The bar does
not move to meet the result.**

**What is open is [piece E](#e-what-c3c-handed-back), and it has three candidate causes rather than
one.** This page said for one revision that the shortfall was a single quantity and that the three
candidates behind it were interchangeable. Neither is right, and the study's own columns are what
say so — [the ranked list](#the-three-candidates-ranked-by-what-the-study-measured) is where each
candidate's evidence and its deciding pull request are.

**There was a fourth candidate nobody had listed, and [E1](#what-e1-landed-and-what-e1b-withdrew)
has removed most of it.** It is the *instrument*: the evaluation draw's own quadrature error lands
directly in every replicate's remainder and is then multiplied by `√n`, so a column reported as
flat and a column too coarsely measured to show a decline look identical. `P₀D̂` is now integrated
deterministically, and at `n = 2,400` the across-draw spread of the column fell from `2.05` under
C3c's rule to `0.33` under the grid, and **[E1b](#what-e1b-measures) has measured what share of that
is the rule**: `0.99` to `1.01` at that size, conditionally identified and with an interval. **Whether any of it resolves the decline
is a rate and neither piece reads one**, in either direction. What it does is make E5's reading of
the same column a reading much more nearly about the estimator.

1. **Theorem fidelity** — the equations solved and the curve reported are the ones the derivation
   gives, under conditions the fit actually meets. Items 1, 13, 15, 21, 22 and 25. The third review
   reported this link **broken** at the sign; reading the source closed item 21 and item 22's
   theoretical half in the implementation's favour, A1a closed item 1, and A1b closed item 15 —
   under a stated entropy condition the shipped libraries meet, with its stability half left to a
   run — so what was left of the link is items 13 and 25 — the empirical remainder rate, whose
   **instrument** landed with [C2](#what-c2-landed) and whose number [C3c](#what-c3c-measured) has
   now read, and **which fits the guarantee is claimed for at all**. **Item 13 has a number and
   does not close on it**: `√n R_remaining` is flat at `1.43 / 1.26 / 1.25` in `q-drift` and does
   not fall in `g-drift`, over two seed batches, so at these sizes and at this reduction the fits
   are outside the assumption Theorem 1's expansion needs. That is a finding rather than a defect,
   and it is one of the two reasons link 4 below fails.

   **Three things separate that measurement from the condition it is read against, and each is a
   piece-E task rather than a caveat.** *Zero, and it is most of the way discharged*: the number
   was read at a precision that could not tell a plateau from a slow decline, and **at `n = 2,400`
   essentially all of that was the evaluation draw** — [E1](#what-e1-landed-and-what-e1b-withdrew)
   integrated `P₀D̂` on a quasi-random rule, which shrank the column's across-draw spread sixfold,
   and [E1b](#what-e1b-measures) measured the share at `0.99`–`1.01`. The same column at E5 will be
   a great deal sharper, and the reading above stands as a reading about the estimator. *One*: the reductions were fitted by `glm`
   and the row two tables down reads `reduced regressions consistent — unverified`, so the
   theorem's own premise was not established in the run that tested its conclusion.
   *Two*: `R_remaining` is an **aggregate** expansion residual, and Theorem 1 asks for `R_{Q,n}` and `R_{g,n}` separately —
   which stay two `unverified` rows, since `branch_products` reports only each branch's
   second-order half and **refuses the `M` terms** rather than approximating them. *Three*: at the
   largest size the decomposition does not separate the two anyway — `cancel` reaches `1.99x` and
   `branches settled` falls to `192/250`, both now **unread** rather than failing, since a movement
   between two bin counts bounds nothing. So what C3c discharged is the *instrument* question, and
   an aggregate that does not vanish is evidence that something in this configuration does not meet
   the condition rather than a reading of which term. What remains open on this link is
   **item 22's numerical half, on one clause of a frozen rule that every measurement misses in the
   direction favourable to it**, which [B2b](#b2b--the-dispatch-and-what-it-decides) reads out.

   **Item 25 is new with this revision, it is a scope decision rather than a defect, and it was
   found by reading this link's own summary against the concordance's assumption matrix.** That
   summary used to say the link was down to items 13 and 15. The matrix does not: `hard truncation
   of ĝ` reads *not covered by the source*, and `B_{A,n}` reads met **under the stated restriction
   that the mechanism is the truncated one and not the theorem's untruncated `g*`**.
   [B1b](#b1b--the-theorem-conforming-targeting-decision) made the solved score and the reported
   curve one expression at one state, which is what it was chosen for and is worth every digit it
   bought — and it is *internal coherence*, not the claim that Theorem 1's expansion applies to the
   constrained estimator. Calling the constrained update a **finite-sample rendering** names the
   step honestly and supplies no such claim. So the link is short by one sentence — *which fits the
   guarantee is claimed for* — and that sentence belongs before [C](#c-the-demonstration) rather
   than after it, because a coverage number read under an unstated scope is a number about an
   estimator nobody has named. [The supported contract](#the-supported-contract-and-item-25) is
   that sentence and what it costs.
2. **Derivation-anchored correctness, component by component** — each object the curve is built
   from agrees with what the derivation gives for it, not merely `psi` and `se`, where several
   differences cancel. **This link is now closed except for item 13's rate, which the dispatch has
   now measured and which [piece E](#e-what-c3c-handed-back) is what closes.**
   `tests/unit/test_remainder_drtmle.py` is the exact-law arithmetic of the expansion,
   `tests/unit/test_theorem_drtmle.py` pins the package's arrays against Theorem 1's own terms and
   the reported interval against its `σ²_n`, and `tests/unit/test_influence_gateaux_drtmle.py` is
   the piece that was missing: the reported curve's own decomposition, against a **derivative**.
   [The concordance's object table](drtmle/theorem-concordance.md#13-the-object-concordance) no
   longer has a `TODO` in its `evidence` column.

   **The anchor is the point and the exact law is not enough by itself.** At the truth `Q_r` and
   `g_{r,2}` vanish row by row, so a check that only asks agreement with an exact law is blind to
   a flipped sign — which is what makes item 21 the worked example rather than a footnote. What
   is not blind is a comparison against the theorem's terms at a **nonzero `Q_r`**, or a
   comparison against the Gateaux derivative on a law where one nuisance is **wrong on purpose**;
   the second is A1a's and is the one that reaches the curve rather than its parts. Its own
   fixture makes the point in the other direction: run it at the truth instead and every
   assertion passes under *either* sign, which is that module's required non-failing control.
3. **Numerical validity** — every required score solved to a statistically negligible order,
   *at the arrays the reported curve is built from*, and a fit that fails to say so somewhere a
   reader cannot miss. Items 11, 12, 16 and 20. The qualification is item 20's: a fit can solve
   all three equations to `1e-11` by its own record and still report a curve whose mean is
   `2e-04`, which is a link-3 failure that announces itself nowhere in the loop's own
   diagnostics. Items 11 and 20 turned out to be **the same failure** — the loud version under
   weak overlap and the quiet one on a quarter of ordinary splits — and the sentence in italics
   above is where the trap is, because *the arrays are the same arrays*. What differs is the
   truncation applied to one of them on the way into two different expressions. "Built from the
   same state" is necessary and it is not sufficient; the checkable form of this link is an
   **identity between each recorded score and a recomputation of the term the curve carries**,
   which is [piece B1a](#b1a--the-identity-and-safety-patch). **This link is now closed apart
   from item 12's second half.** B1a made the identity a reported number per arm on the face of
   every fit rather than something inferred from an uncentred curve, and
   [B1b](#b1b--the-theorem-conforming-targeting-decision) made it hold: the score is solved at the
   truncated tilt the curve reads, and items 11 and 20 close together because they were always one
   failure. What the instrument bought is that the closure is a *measurement* — the same rows, the
   same `1e-12` bar, verdicts the other way up — rather than an argument that the new expression
   must be right.
4. **Inferential usefulness** — coverage in a regime where the plain interval fails. Item 3, and
   [C3c](#what-c3c-measured) has read it. **The improvement is real, large and reproduced; the
   coverage is not nominal.** `TMLE` `0.532` / `0.472` against `DRTMLE` `0.844` / `0.848` at
   `n = 2,400` in `q-drift`, paired `+0.312 ± 0.031` and `+0.376 ± 0.033` — and `DRTMLE`'s own
   interval misses `0.95` in every cell at every size. So this link is **half closed**, and the
   half that is not is **two measured shortfalls with one open premise behind them**, which is a
   revision: three revisions of this page said *two mechanisms* and attributed the second to the
   variance estimator. `q-drift` at the largest size misses nominal by eleven points and its
   `se ratio` is `0.903`; both are real, both reproduced to the digit, and **neither has a repair
   that is not item 13's** —
   [the withdrawal](#the-se-shortfall-is-a-symptom-and-what-withdrawing-it-costs) is where the
   arithmetic and the evidence are. What is licensed is the *ranking*: the bias term is the larger
   under either ordering of the two counterfactuals, and the two are not additive. In
   `g-drift` at the two smaller sizes the improvement is *negative*, which the regime-entry column
   explains — the correction removes 3–8% of the targeted remainder there — and its `se ratio` is
   `1.157`, the other side of one, which is the observation that costs the variance-defect reading
   its mechanism.

The first review's summary of this is exactly right and worth keeping in its words: none of the
four implies the others. A curve that matches the theorem proves nothing about whether the
alternation solved its equations on a given draw; a solved fit proves nothing about whether the
interval is ever *better* than the one `TMLE` already reports; and a coverage number proves nothing
about which of the two the estimator got right.

**There were five links, and the one that has gone was `drtmle` parity.** It has been [retired by
decision](#closed-since-this-list-opened) — item 2 — and this page had already written the reason
twice before acting on it: both packages descend from one source, so agreement is evidence about a
transcription and is blind to exactly the class of error item 21 is in. The corollary that cost the
most to learn is that **that blind spot is shared with the exact law**, which is why link 2 above
carries its anchor in italics: two checks that cannot fail against the same class of error are one
check, however different their machinery
([lesson 9](drtmle/investigation-log.md#what-the-sizings-got-wrong)).

**A1a is that lesson applied rather than restated, and it is worth saying how.** Its instrument is
a Gateaux derivative, which is different machinery from the theorem-terms comparison — but
different machinery is not the test. The test is whether the two can fail against different
things, and they can: the theorem comparison reaches the corrections' *parts* at a nonzero `Q_r`
and is silent about what a fit assembles from them, and the derivative reaches the assembled curve
from a real fit and is silent about the parts. Each was watched to fail against a library mutation
the other does not see. And the module's own docstring names **four things it cannot see at
all** — two of them measured by running the mutation and watching it pass — which is the half of
this lesson that never gets written down, because a suite records what it caught and not what it
cannot. One of the four is item 15, and it is why [stop-ship 14](#stop-ship) is new with this
piece.

The [limitations](#limitations-recorded-rather-than-fixed) after the pieces are outside the chain
entirely: real, understood, and unable to move a coverage number. Anything that *can* move one
belongs in a piece, which is where item 20 went after being filed there by mistake — see
[lesson 7](drtmle/investigation-log.md#what-the-sizings-got-wrong).

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
`estimators/targeting.py` and `tests/unit/test_drtmle_fit.py` all cite them by number. **The first
review's items are therefore 13 to 19 rather than a renumbering**, item **20** — found while
closing item 18 — is 20 for the same reason, the third review's two are **21** and **22**, which is
why the most important item on the page has the highest number, and **23** was found by piece
B1a's own instrument on its first run and is now closed. The pieces are lettered so the two cannot
be confused.

### The work, in five pieces and eighteen pull requests

A, B, C and E are each split, so the five pieces are eighteen pull requests: **B1a**, **A1a**, **B1b**,
**B2a**, **B2b**, **A1b**, **C1**, **C2**, **C3a**, **C3b**, **C3c**, **E0**, **E1**, **E1b**,
**E2** and **E2R** have landed — **E2 has run and did not branch**, which is a result and not a
deferral, and **E2R's instrument is built with its decision run owed**, which is the C1/C3a/B2a
pattern and not a deferral either — and **D** and the rest of **piece E** are open. **The count is
unchanged and its composition is not**: piece E's tail is now [a diagnostic, one branch and a
study](#the-reboot-and-the-three-work-packages), so **E2R** is a row that did not exist and
**E2b** and **E3** are alternatives of which at most one is ever written.
**C3 split into three after its pilot ran**, under
the same rule as B2: the repair precedes the dispatch and closes nothing on its own. **E is the fifth piece and it
exists because C3c measured rather than settled** — a study whose premise was `unverified` cannot
close the item it read, and the alternative to a piece was leaving that as a paragraph. **The
contract is an eleventh row and is not a piece** — it is documentation, it closes no research, and
it is in the table because it is an input to C that had to be frozen rather than produced; **E0 is
the same kind of row for the same reason**. Small
items are grouped where the *evidence* is shared — B2 is five items because one dispatch of the same
sweep answers all of them — not where the subject matter merely rhymes; and the four splits are
that same rule applied to a piece rather than to an item. **B2's grouping was right about four of
its five and wrong about the fifth**: the exit distribution, the closing cap, the weak-overlap
policy and the overlap attribution all fell out of one dispatch as planned, and the update-order
question needed three more and still did not close.

**C splits three ways under that same rule, and the split is the B2a/B2b one twice over.** Tier 1's
evidence is the exact remainder of a *prescribed* nuisance sequence — a quadrature, with no learner
and no fold draw in it — and Tier 2's is a coverage gap under estimated ones, which needs the
fold-retained nuisance object `P₀D̂` cannot be computed without. Those share nothing but a harness.
And the dispatch shares nothing with either: it is the one run on this page whose *cost* makes
redoing it a decision rather than an errand, so its inputs are frozen before it and not during it.
So **C1** is the instrument, **C2** is Tier 2 and item 13's columns, and **C3** is the study and
item 3 — which the pilot then split again into **C3a** the freeze-and-pilot, **C3b** the repair its
numbers forced, and **C3c** the study itself. **E is the same rule applied once more**: the study
returned a number that does not close the item it read, so the diagnosis splits from the dispatch
exactly as the repair did. **Two of its rows cost a run** — E2R the cheap diagnostic one and E5 the
expensive confirmatory one — and the whole point of the ordering is that the second waits on the
first.

| PR | what it lands | new artefacts |
| --- | --- | --- |
| **B1a** — *landed* | the score/correction identities, the clipping diagnostic, and invalidation when either fails | `cleverly/validation/drtmle.py`; `res.validation.correction_check()`; `identity` and `correction` rows on `score_check`; tests in `tests/unit/test_drtmle_fit.py` and `test_influence_drtmle.py` |
| **A1a** — *landed* | items 1 and 21, and item 22's theoretical half: the theorem read, mapped, graded, the sign adjudicated, every object pinned to the test that derives it, and the decomposition pinned against a perturbation of the law. Item 22's *numerical* half — both orders on real data — went to B2, whose dispatch it shares | `tests/unit/test_influence_gateaux_drtmle.py`; `TestTheReportedVarianceIsTheorem1s`; closes out [`docs/drtmle/theorem-concordance.md`](drtmle/theorem-concordance.md) — its object table's `evidence` column, its assumption matrix, and §7's finding for B1b |
| **A1b** — *landed* | item 15: the argument for the pooled construction **and** the reference estimator that measures its open condition, which turned out to be one deliverable rather than two. The entropy half closes on the learner; the stability half is **supported and not shown**, and the dispatch that read it is the second median-based clause at twelve draws to fail to carry a slope | `DRTMLE(reduced_crossfit="nested")`, `InnerDesigns`, `Fluctuation.carried` and `MechanismFluctuation.carried`; `tests/unit/test_nested_reductions.py`; `--reduced-crossfit` and a workflow input; [the concordance's §8](drtmle/theorem-concordance.md#8-cross-fitting-is-not-in-the-sources-and-the-argument-for-it-item-15) rewritten around the argument and its six new matrix rows; [the plan's §7](drtmle/validation-plan.md#7-the-cross-fitting-construction-piece-a1b) |
| **B1b** — *landed* | items 11 and 20: the targeting convention, chosen on theorem fidelity against a fitted prototype that eliminated two candidates by construction and separated the other two by measurement | `solve_bounded_mechanism`, called from the two `DRTMLE` sites, with the truncated array carried forward and `"bounds_pinned"` where no constrained root exists; `tests/unit/test_bounded_mechanism.py`; `CorrectionRow.margin` in place of B1a's now-vacuous clipped-row witness |
| **B2a** — *landed* | the instrument the dispatch needs: the columns §4 asks for, the working paper's update order beside this one, and the comparison arms. Closes nothing on its own, which is why it is its own pull request rather than a first commit of B2 | `DRTMLE(update_order="paper")` and `ReductionSpec.order`; three tables on `benchmarks/bench_drtmle.py` in place of one; `--order`, `--reduced-learner` and `--truncation` arms with workflow inputs; `TestBothUpdateOrdersReachTheTheoremsExit` |
| **B2a′** — *landed* | the three things B2a left as prose rather than as something a run settles: the oracle reduction built where it exists, a **control** for the update-order difference, and the branch that hid a mutation deleted rather than guarded | `tests/unit/test_oracle_reductions.py` and item 24; the `reseed` arm, the route-against-noise table and [a frozen rule](drtmle/validation-plan.md#the-update-order-rule-frozen-before-the-dispatch); `_restated_outcome_score` made unconditional, `tests/unit/test_fluctuation_score.py`, and [lesson 12](drtmle/investigation-log.md#what-the-sizings-got-wrong) |
| **B2b** — *landed* | items 12 and 19, re-measures 4 and 6, takes the overlap policy — no refusal — and reads item 22's numerical half, which stays open on one clause | four dispatches of `drtmle-convergence.yml` and their tables in [the investigation log](drtmle/investigation-log.md#what-the-b2b-dispatch-measured); `_negligible_bar`; a `worst identity` column on `comparison_rows` and `tests/unit/test_bench_drtmle.py` |
| **the contract** — *documentation, and it has landed* | item 25: which options the theorem-backed guarantee is claimed for, and which are supported beside it. Not a piece and not research — the scope decision, its conditions as assumption rows, and the two review readings it corrects | [the table](#the-supported-contract-and-item-25); [the concordance's §7 scope decision](drtmle/theorem-concordance.md#the-scope-decision-item-25) and three new matrix rows; gate 1's clause 0. Its per-fit witness landed with C1 and **found the contract's condition is not the ordinary case it was read as** |
| **C1** — *landed* | the study's instrument and Tier 1 complete: prescribed nuisance sequences with their drift coefficients committed analytically and verified by quadrature, the paired harness, the dispatch workflow, and item 25's per-fit truncation witness. Closes no numbered item on its own, which is why it is its own pull request rather than a first commit of C | `benchmarks/drtmle_injection.py`, `benchmarks/drtmle_coverage.py`, `.github/workflows/drtmle-coverage.yml`, [`docs/drtmle/coverage-study.md`](drtmle/coverage-study.md); `CorrectionCheck.contract` with `initial_clipped` and `gr1_margin`; `DGP.expectation`; `tests/unit/test_drtmle_coverage.py` and `TestTheContractSaysWhichEstimator` |
| **C2** — *landed* | item 13's **instrument**: Tier 2's prescribed-rate learners, the evaluation companion `P₀D̂` needs, and the two appendix branches reported apart. It does not close item 13, which is a *rate* and so C3's dispatch's | `DRTMLE(evaluation=…)`, `CompanionEstimates`, `cross_fit_companion`, carry-with-its-own-covariate in both fluctuation solvers; `benchmarks/drtmle_tier2.py` and `benchmarks/drtmle_remainder.py`; `--tier 2` and `--evaluation-n` on the harness and the workflow; `tests/unit/test_drtmle_companion.py`, `test_drtmle_tier2.py`, `test_drtmle_remainder_study.py` |
| **C3a** — *landed* | the freeze and the pilot: the mixed-cell rule, the two gate-1 clauses that could not be read at all, the invalid-fit threshold frozen where it was, and **the pilot that falsified the sizing it was run to check**. Closes no numbered item, which is why it is its own pull request | four dispatches of `drtmle-coverage.yml`; [what the pilot measured](drtmle/coverage-study.md#what-the-pilot-measured); `benchmarks/drtmle_tier1_bias.py`; §5's fourth operational rule and its new targeted-coefficient clause; `identity`/`score`, `cancel`, `sqrt(n) R2` and the stratum table |
| **C3b** — *landed* | the repair the pilot forced: a **second declared coefficient** — the estimator's bias, not the plug-in remainder — and a Tier-1 injection built to hit it, plus the regime-entry column Tier 2 never had and the three pre-flight conditions as a verdict table | `targeted_coefficients`, `targeted_weight`, `exact_targeted_remainder`, `population_epsilon` and `Q_DRIFT_B`/`G_DRIFT_B_ATE` on both tier modules; `drtmle_remainder.targeted_remainder`; `entry_rows` and `preflight_rows` on the harness; the decomposition tables on `benchmarks/drtmle_tier1_bias.py`; [the repair](drtmle/coverage-study.md#the-repair-and-what-would-say-each-half-of-it-is-wrong) |
| **C3c** — *landed* | **item 3, answered**: the study entered its regime for the first time in three attempts and produced the gap — `+0.312 ± 0.031` and `+0.376 ± 0.033` paired in `q-drift`. **Item 13 gets its first number and stays open**: `√n R_remaining` does not vanish at these sizes *at `glm` reductions*, which is a measurement of this configuration rather than of the theorem's condition | two dispatches of `drtmle-coverage.yml` at 250 replicates and their tables; per-replicate results as four artefacts, now [manifested](drtmle/study-manifest.md); [what it measured](drtmle/investigation-log.md#what-the-c3c-dispatch-measured); [the gate readout](drtmle/coverage-study.md#the-gates-read-out-clause-by-clause); lessons 17 and 18 |
| **E0** – **E1b** — *landed* | [what C3c handed back](#e-what-c3c-handed-back)'s first three: the record, and the remainder instrument. E1 integrates `P₀D̂` deterministically, which takes most of the instrument off the list of candidate explanations for the flat column. **Its two readings of *how much* are withdrawn** and are E1b's | `DGP.quadrature`; `quadrature_frame`, `truth_at`, `corrected_curve` and `row_weights`/`limit` on every remainder column; `benchmarks/drtmle_companion_grid.py` and its workflow; `--quadrature-points`; `ScoreRow` and a second artefact; `tests/unit/test_drtmle_companion_grid.py` |
| **E1b** — *landed* | the same question measured rather than asserted: an independent scramble per replicate, so the grid's error is mean-zero and estimable *conditionally on each fit*, with an interval on every share. **At `n = 2,400` the draw accounted for `0.99`–`1.01` of the column's across-draw variance** | `DGP.quadrature(scramble=…)`; stacked companions and a row `Window`; `rule sd` and `share`; two dispatches, [manifested](drtmle/study-manifest.md#e1b-what-was-run) |
| **E2** — *landed, and it did not branch* | the reference reduction dispatched against its frozen rule. **Gate B fails in three cells of four**, each on a coarser rung beating the shipped one, so those cells are `unresolved`; the fourth reads **`moved`**. Candidate 1 is alive-but-unestablished rather than decided, and the named repair is a rung **selected rather than shipped** — [at interval level a rung per *cell*](#what-e2-measured-and-why-it-did-not-branch), which is a correction to two earlier revisions | run `31042558057`, four artefacts [manifested](drtmle/study-manifest.md#e2-what-was-run); [what it measured](drtmle/investigation-log.md#what-the-e2-dispatch-measured); `benchmarks/drtmle_reference.py` and `drtmle_reference_study.py`; `.github/workflows/drtmle-reference.yml`; [§8](drtmle/validation-plan.md#8-the-reference-comparison-piece-e2); `tests/unit/test_reference_exact_law.py` and `test_drtmle_reference_study.py` |
| **E2R** — *the instrument has landed; the decision run is owed* | the reference repaired at [eight points](#the-reboot-and-the-three-work-packages) and its rule frozen: the rung **selected** per `(cell, size, regression)` against a measured ranking, four blocks so the block that certifies is not the block that chose, **five** gate-B metrics — the three regressions and the two composites the correction divides by — a control coarse enough to be rejected on all five, and the block-size falsifier declared. A review of that instrument then found three ways it could still pass without certifying anything, so the last three are the **decision protocol**: two disjoint cohorts of draws with a committed manifest between them, a fidelity clause that is non-inferiority against a declared margin rather than failure to show superiority, and a run that exits non-zero on an incomplete artefact set. Closes no numbered item on its own, which is why it is its own pull request; the run dispatches from `main`, as E2's did | `benchmarks/drtmle_reference.METRICS`, `composite_denominators`, `metric_weights`, a per-regression `reference=`; `select_rung`, `RecordingDRTMLE`, a selection table and a third artefact on the harness; `cohort_seeds`, `SelectionManifest`, `validate_selection`, `noninferiority_margins`, `simultaneous_lower_bounds`, `run_integrity`; [§8's selection rule and what the pilot moved](drtmle/validation-plan.md#the-selection-rule-frozen-before-the-dispatch) and [its decision protocol](drtmle/validation-plan.md#the-decision-protocol-frozen-before-the-dispatch); `tests/unit/test_drtmle_reference.py` and `test_drtmle_reference_study.py` |
| then **E2b** *or* **E3**, then **E4** + **E5** | [the three work packages](#the-reboot-and-the-three-work-packages): **one** branch off E2R's verdict, and the frozen confirmatory study. **E5** is where item 13 closes either way | their rows below |
| **D** | the two candidates in item 10 | its own reduced object, submodel and fixtures |

**A1 split into A1a and A1b for the reason B1 split into B1a and B1b**, and the reason is worth
having written down because both splits look like scope management and neither is. B1 split
because one half preceded the other and depended on it. A1 splits because **item 15 shares
evidence with nothing else in the piece**: every other row of A1 was a test to write or a reading
to record, and item 15 is a proof to find or a second estimator to build. Grouping them would have
made the half that unblocks B1b wait on the half that unblocks nothing. That is the same rule the
pieces were lettered under — *grouped where the evidence is shared, not where the subject matter
rhymes* — applied to a piece rather than to an item.

**The dependency order is the plan, and it is not the reading order below.**

```text
B1a  identity + safety patch ─────────────────────┐   landed
                                                  ├─> B2a ──> B2b ──┐
A1a theorem concordance ──> B1b  targeting  ──────┘  instrument  sweep │
                                 convention          landed      landed │
                                                                        ├─> C1 ──> C2 ──> C3a ──> C3b ──> C3c
contract + item 25  scope frozen ──────────────────────────────────────┤  harness  tier 2  freeze   repair  study
A1b cross-fitting construction ────────────────────────────────────────┘  landed   landed  + pilot  landed  landed
       (no logical block; a rework edge — C's fits are of A1b's construction)   landed          │
                                                                                                │
C3c ──> E0 ──> E1 ──> E1b ──> E2 ──> E2R ──> [ E2b | E3 ] ──> E4 ──> E5  <────────────────────────┘
       record  P₀D̂   rule's  refer- selected  one branch      two     fresh
       landed  rule   error   ence  rung +    of the two,     designs dispatch
              landed  landed  ran,   4 blocks never both
                              no     built,
                              branch run owed
                                     ─ WP1 ─  ─── WP2 ───     ──── WP3 ────

       E2b fires if the repaired reference improves the remainder and a feasible learner is
       in reach; E3 if it does not, or if a repaired remainder still leaves the calibration
       failing. A reference that stays unresolved, or a gap no feasible learner closes, stops
       the release instead of advancing to the next box

D   independent of all of it, and gated on A1a alone. Not a fallback inside this release
```

**Every piece of the graph except D and WP2 onwards has landed, and the demonstration is *made and
not passed*.** E2 is the one that landed without deciding what it was built to decide: it ran, its
gates held it to `unresolved` in three cells of four, and
[what it measured](#what-e2-measured-and-why-it-did-not-branch) is the finding — a reference whose
best resolution is not the one that was shipped. **E2R is the repair to that, and what has landed
is its instrument**: the rung selected rather than shipped, four blocks so that the block which
certifies is not the block which chose, five gate-B metrics where there were three, and a control
coarse enough to be rejected on all five — [with two of the five clauses moved by the pilot that
sized them](#what-e2rs-instrument-landed). Its decision run is owed and dispatches from `main`. **E3 has moved from the trunk onto a branch**, and
that is the reboot: it was going to run whatever E2 said, on a `se`-ratio mechanism
[this revision withdraws](#the-se-shortfall-is-a-symptom-and-what-withdrawing-it-costs).
[C3c](drtmle/coverage-study.md#what-the-study-measured) entered the regime, produced the gap, and
read the gates out: gate 1 fails at four clauses and gate 2 passes its shortfall clause and fails
its invalid-rate one. What is left is [piece E](#e-what-c3c-handed-back), whose question is not
*"whether `√n R_remaining` can be made to vanish"* — that was one revision's phrasing of it and it
assumed the answer lay in `n`. It is **which of three candidates the flat remainder is**, and the
first of them is testable for the cost of a quadrature.
The C1 → C2 edge was the harness and the injection interface, which Tier 2 extended rather than
replaced; the C2 → C3a edge — that a dispatch whose remainder columns are missing cannot read gate
1's clause 4 — is **discharged**: the columns exist and both tiers dispatch from the one workflow.
**The C3a → C3b edge was a real block rather than a rework one and it is now discharged**: C3a's
pilot measured a design that does not enter the regime it committed to, so C3c's dispatch would
have answered for an estimator nobody had characterised. C3b's repair lands and its pre-flight
reads **conditions 1 and 2 passing in all four cells**, with condition 3 unresolved at 12 draws
rather than failing — and the dispatch then read that condition at 250 draws in both cells, where
it is **flat rather than falling**. That was the block being real: the repaired design entered its
regime and the pilot's would not have.
Two open threads sit behind it rather than pieces. Item 22's numerical half is answered
on `nonlinear` and unresolved on `weak-overlap`, and it gates [stop-ship 2](#stop-ship) rather than
gating C. And A1b's construction decision rests on its argument's **entropy** half, which the
learner settles, plus its **stability** half, which a dispatch reads — so the rework edge into C is
narrower than it was but is not gone: if the nested arm's difference does not shrink,
`reduced_crossfit=` changes default and C's fits are of the other construction.

**Two edges into C are new with this revision and neither is a logical block.** They are there
because C is the one dispatch on this page whose *cost* makes redoing it a decision rather than an
errand — ~2,000 fits, re-timed but still a workflow of its own. The
[contract](#the-supported-contract-and-item-25) is an edge because a coverage number read under an
unstated scope is a number about an estimator nobody has named; it is a page of prose and freezing
it costs nothing. **A1b is an edge because the estimator C fits is the pooled construction**: if
A1b concludes that construction has to change, C's numbers are numbers about the wrong estimator
and the study is rerun. **That edge is now discharged** — the decision is frozen at pooled — but it
is not gone, since the condition it rests on is supported rather than shown. This page said A1b "blocks nothing", which is true of every *other* piece
and was read as true of C — and A1b's own section already says the outcome may be that a nested
reference implementation is needed. That is a rework edge, and the ordering rule it implies is
narrower than "finish A1b first": **A1b's construction decision must be frozen before C's final
dispatch**, which a pilot of 50–100 replicates does not wait on. Ordering the whole demonstration
behind an open research question would be the opposite mistake.

**B1a first**, because every number B2 and C produce is read *through* the reported curve, and
until it lands a share of every cell's fits report a curve the fit did not solve for. It is also
the cheapest thing on this page, and — the point of splitting it out — **it is valid under every
convention B1b might eventually choose**, so it does not wait on the theorem.

**A1a alongside it**, outranking everything because if the theorem and the transcription disagree,
work already landed is work to redo — and on the sign they already did. **B1b after A1a**, because
the convention is a derivation and not a taste. **B2** on the corrected implementation. Then the
**contract** — a page of prose, and the cheapest thing left on this list — and **A1b**, whose
construction decision has to be frozen before C's final dispatch even though nothing waits on it
and even though its pilot does not. **C** last, because it is the expensive one and the only one
whose inputs are worth freezing on purpose. **D** independent of all of it.

The graph used to have a second arm into B1b — the R-parity piece, item 2, [retired by
decision](#closed-since-this-list-opened) — and before that it had B1 both first and downstream of
that piece, which is not a dependency graph. Splitting B1 is the third review's most useful
structural correction; dropping the parity arm is what this page's own argument had implied for two
revisions.

#### A. Check the curve against something other than itself

**Closes items 1, 15, 21 and 22, and opens item 13.** The influence curve
`D = D* − D*_Q − D*_g` was read off `drtmle`'s implementation, not derived. The whole variant is a
variance estimate, so a curve transcribed from software and never checked against its derivation
is the one part of this that could be wrong in a way nothing here would catch — and
[it was](drtmle/theorem-concordance.md#4-the-sign-discrepancy-item-21--resolved), on a sign.

**It is checked now, twice and against different things**, which is the piece's whole point:
against Theorem 1's appendices at a nonzero `Q_r`, and against the Gateaux derivative of the
parameter on a law where one nuisance is wrong on purpose. Two checks blind in the same place
would have been one check; these are not, and the second is the one that reaches the *curve*
rather than its parts.

**Two pull requests, [A1a](#a1a--the-theoretical-audit) and
[A1b](#a1b--the-cross-fitting-construction), and A1a has landed.** What is left of the piece is
item 15 alone.

**One piece, where there used to be two.** The second was `drtmle` parity — item 2 — and it is
[retired](#closed-since-this-list-opened) rather than done. What it was right about was the
*decomposition*, and that has moved here: the components are checked against the derivation
instead of against another implementation.

**The document-access problem is gone and was never a paywall.** The 2023 software article and the
**2016 Berkeley working paper** were both supplied by hand and both read first-hand rather than
through a transcription — which is what closed item
21, and which changed the answer: the transcription was faithful and the display it transcribed is
contradicted by the same paper's appendices. What remains unobtained is the **published 2017**
article, an NIH author manuscript in PubMed Central as **PMC5793673**, and van der Laan (2014)
Theorem 3, which only [piece D](#d-widen-the-scope-to-what-the-sources-derive) needs. Neither now
gates anything: item 21 was settled on internal consistency plus exact-law arithmetic, and neither
depends on the edition. One runner's network measurements are in [the investigation
log](drtmle/investigation-log.md#what-one-runner-could-and-could-not-reach) rather than here,
because they are a property of an execution environment on a date and this page carried an
obstacle it had inherited for two revisions.

**The two files themselves are no longer in the repository**, and neither is anything they were
cited for: what was read out of them is transcribed into
[the concordance](drtmle/theorem-concordance.md), which now carries a **page number** on every
section, display and appendix it quotes. That is what the file paths were standing in for and
doing badly — a path resolves for a reader who already has the file, and a page number resolves
for one who does not. [`references.md`](references.md) is where every source in the package is
listed.

##### A1a — the theoretical audit

*Closes items 1, 21 and 22.* **Landed.** Items 21 and 22 closed on reading the working paper;
item 1 closed here. Item 15 is [A1b](#a1b--the-cross-fitting-construction).

- ~~**Adjudicate the sign (item 21) first.**~~ **Done, and it did not need the published
  article.** The plan was to obtain the published 2017 Theorem 1 and see whether it retained
  `D_A = −(Q_r/g)(A − g)` with `D^{*,#} = D* − D_A − D_Y`. What settled it instead was reading the
  working paper's **appendices**, which derive each block as
  `P_0[term] = −(P_n − P_0)D + P_n D + (second order)` — an identity satisfiable only with `D`
  equal to the *positive* correction. So `D_A = +(Q_r/g)(A − g)`, Theorem 1's `σ²_n` is what this
  package computes, and the §3.1 display is wrong on its face in a paper that also prints `D_Y`
  twice with two signs. The fixture the plan asked for exists as
  `tests/unit/test_theorem_drtmle.py`: nonzero `Q_r`, the appendix step that fixes the
  orientation, the representation closing to `1e-12` with the corrections subtracted and failing
  by *exactly twice the correction* when added, the two readings' variances separated, and the
  package's arrays pinned against the theorem's terms. **Nonzero `Q_r` is the load-bearing word
  there**: at the truth both corrections vanish row by row, so a comparison taken at an exact law
  agrees with a flipped sign and this module is the one instrument here that does not.
- ~~**Close out the concordance.**~~ **Done.** The permanent table maps each object of the theorem
  to its Python name and its R name, and states for each: the conditioning variable, the sign, the
  denominator and its truncation, whether the value is initial or starred, whether it is
  arm-specific, **which score or influence term consumes it**, and — the column item 2's
  retirement is traded for — **which test pins it against the derivation**. `TODO` is gone from
  that column. Two rows are open and each now says *whose*: `R_{Q,n}`/`R_{g,n}` is item 13 and
  piece C's, and the reduced regressions' cross-fitting is item 15 and A1b's. Writing "open, owned
  by C" is not a downgrade of the bar — `TODO` said *nobody has looked*.
- ~~**Pin the curve's decomposition against a perturbation of the law.**~~ **Done, and it took a
  different shape from the one planned.** The plan was a further `discrete_law*` module carrying
  the whole DRTMLE limit as an analytic functional — initial nuisances, reductions, the
  alternation — differentiated by complex step. It does not need one, because **in the union model
  at saturated reductions the corrected curve collapses onto the efficient influence function**:
  `1/g_1 − g_{r,2}/g_{r,1} = 1/g_0` on one side, and on the other the `Q̄*` in `D*` cancels
  against the one inside `Q_r = Q̄_0 − Q̄*`. So the derivative already in the repository —
  `tests/discrete_law.py`'s `eif` — is the right-hand side, and what A1a had to write is the
  left. `tests/unit/test_influence_gateaux_drtmle.py` is that, in two tiers: this module's own
  longhand, and **a real `DRTMLE` fit**, which is also the first fit in this repository against a
  deliberately misspecified law. Both close at `~1e-15`; a flipped sign misses by `0.55` to `2.8`.

  Three things it deliberately cannot see are named in its docstring, each **measured by running
  the mutation and watching it pass**: item 23, equation (9)'s covariate sign, and a reduced
  regression's pooling weight. All three are one degeneracy — a cell is blind to every mutation
  of a term it sets to zero — and each names the module that does cover it. That list is the
  point rather than an apology: an instrument whose blind spots are unlisted is how [lesson
  9](drtmle/investigation-log.md#what-the-sizings-got-wrong) happens again.
- ~~**Finish the assumption matrix**~~ — **done**, one row per condition, with columns
  `condition | source | required for | what the implementation does | evidence | status`, and a
  status drawn from *met*, *met under a stated restriction*, *unverified*, *violated*, *not
  covered by the source*. The six empty `evidence` cells are filled and **the count of
  `unverified` rows did not fall**, which is what the column is for: a row now says what would
  settle it rather than nothing at all. Two corrections: `B_{A,n}` read *violated until B1 lands*
  and B1a has landed, so it now reads violated-**and-measured**, per arm, with B1b closing it; and
  the count of rows reading *not covered by the source* was three on this page and is four —
  missing outcomes was added and this page had not caught up. **It is five since item 25**, which
  split the truncation row in two: `ĝ`'s truncation has an assumption in the theorem to be scoped
  against and `g_{r,1}`'s has none, and the matrix had been carrying them as one.
- **Settle the update order (item 22) — the theoretical half is closed and the numerical half
  moves to [B2](#b2--the-sweep-on-the-corrected-implementation).** The theorem states a six-step
  recursion in a particular order and the Python iteration is not a transcription of it; the
  paper's step 7 states its own exit as the three score equations, so the order is not
  prescriptive and the fidelity question is answered. What is left is the reassurance that the two
  routes reach the same fixed point *on real data*, and that is a second alternation run over a
  sweep of draws — it shares its evidence with B2's dispatch and shares none with anything else in
  A1a, which is the grouping rule this page is lettered under. **This is a move and not a
  downgrade**: it stays on the page, it stays numbered, and [stop-ship
  2](#stop-ship) still names it. When it runs: implement the paper's order beside this one and
  compare the fixed point each reaches and the final three theorem-defined scores at each —
  **both orders here**, since what is in question is two routes to one stated exit rather than two
  implementations, and the comparison wants the same nuisances on both sides. **Do not compare
  fluctuation coefficients across algorithms** unless the submodels and the order are identical.
- ~~**Answer the question B1b waits on.**~~ **Done, in [the concordance's
  §7](drtmle/theorem-concordance.md#7-truncation-is-not-in-the-theorems-algorithm)**, which now
  states it as a finding rather than leaving it as a reading: *the theorem's `D_g` is evaluated at
  the same `g*` its score is solved at, and that `g*` is not truncated anywhere.* The consequence
  changes what B1b is. The theorem has one mechanism, not two conventions to choose between — so
  B1b is not adjudicating a reading of the source but choosing a **finite-sample rendering of a
  step the theorem states without one**, against the bar that the final score be the
  theorem-defined score of the estimator declared. Nothing in the sources will settle that, which
  is why no further document is owed and why B1b was never really waiting on one.
  **What that bar does not do is say which estimator is declared**, and this page went two
  revisions without noticing the difference: a rendering whose final score is its own
  theorem-defined score is internally coherent, and Theorem 1's *expansion* is still stated for the
  untruncated one. [Item 25](#the-supported-contract-and-item-25) is the missing sentence, and it
  is a scope decision rather than a reopening of B1b.

Two of the assumption rows were open questions before any of this and are numbered:

**13. The theorem asks for one thing more than the three score equations, and it is unmeasured
here.** Solving equations (8), (9) and (10) is *necessary*; Theorem 1 separately assumes the
remaining second-order terms are `o_p(n^(−1/2))`. Nothing on this list checked that, and coverage
in piece C could come out right without it — which would be an accident nobody could distinguish
from a result. Half the check exists already: `tests/unit/test_remainder_drtmle.py` shows on an
exact law that one guard removes the whole first-order part of `R₂`, that the unguarded remainder
is not already zero, and that coarsening a reduction leaves a residue — the arithmetic, at
saturated reductions. The missing half is empirical, at *estimated* reductions:

```text
R_remaining = psi-hat − psi_0 − (P_n − P_0) D-hat_DR
```

computed at a known truth and shown to satisfy `√n · R_remaining → 0` across sizes in **both**
off-diagonal regimes. That is a column on piece C's study rather than a run of its own, since that
study is the only place that knows `psi_0` and already fits both estimators at three sizes — so
item 13 is opened here, where the reason for it is, and closes at
[E5](#e-what-c3c-handed-back) — **not** at C3c, which measured it. **It is now known to be
insufficient on its own**: the working paper's appendix B splits the remainder into `R_{Q,n}` and
`R_{g,n}`, and a total trending to zero can conceal cancellation between them, so the study
reports the branches separately where the DGP permits. **That caveat is no longer hypothetical.**
C3c's `cancel` column reaches `1.99x` in `g-drift` at the largest size and `branches settled`
falls to `192/250`, so at the size that matters most the decomposition was still moving between
the two bin counts — and what the branches report is each one's *second-order half*, since `branch_products`
refuses the `M` terms rather than approximating them. A total that does **not** vanish is
therefore evidence that something in the configuration fails the condition, and not yet a reading
of which term does. The exact terms are in [the concordance's
§5](drtmle/theorem-concordance.md#5-the-remaining-remainder-terms).

**The column exists as of [C2](#what-c2-landed) and the item does not close on it.** What that
piece built is the thing `P₀D̂` was missing — the fit's own nuisance functions at rows it never
saw — and the two exact columns beside it. What it did not do is *run* three sizes at 250
replicates, which is C3's dispatch and is where a **rate** can be read. An instrument landing is
not a condition being met, which is the rule [A1b's stability half](#what-a1b-landed) is held to
and this is held to the same one.

##### A1b — the cross-fitting construction

*Closes item 15, and it is the whole of the piece.* **Blocked by nothing**, which is why it is its
own pull request rather than half of A1a. [Gate 1](#c-the-demonstration) does not open without it,
so it is not optional.

**It used to read "and it blocks nothing", and that was wrong about one edge.** Nothing *waits* on
it in the sense every other piece here does — no piece cannot start until it lands. But the
estimator [C](#c-the-demonstration) fits is the pooled construction, and the two live outcomes
below are not symmetric: "pooled is fine, and here is why" costs a paragraph and leaves C's numbers
standing, while "it is not" makes a nested construction the reference and turns a completed
coverage study into a study of the wrong estimator. A dependency that only bites in one branch is
still a dependency, and the cost of the branch it bites in is the whole of piece C. So the rule is
**A1b's construction decision frozen before C's final dispatch** — not A1b proved before C begins,
which would put the demonstration behind an open research question.

**15. The reduced regressions' cross-fitting is defended in an implementation note, not an
argument.** `fit_reduced`'s docstring is the most careful writing in this variant and it reaches
the right conclusion for the wrong kind of reason: it reuses the primary split, shows that an
independent split removes *none* of the induced dependence (the contamination is in the design
values, not in which rows train), and shows that per-fold designs would trade a second-order
dependence for a first-order covariate shift. All of that is sound and none of it establishes what
the theorem needs, which is that the induced dependence is higher order in the expansion. **The
working paper does not close this**: it presents no cross-fitted Theorem 1 at all, and the 2023
article's general claim that cross-validation weakens the entropy conditions is supportive and is
not a proof for this pooled construction. The real question is the one the first review framed:
*determine a construction satisfying the empirical-process conditions of the DRTMLE expansion, and
say whether fold reuse is one.* Candidates: nested cross-fitting, three-way splitting,
per-outer-fold reductions, and the current pooled construction with a proof. Agreement with R would
have settled nothing here in any case — that package predates the construction — which is one of
the several reasons item 2 was never going to earn its keep. Keep both tracks: if the answer is
"pooled is fine, and here is why", the docstring gains a
paragraph and the item closes; if it is not, the expensive nested version is the reference
implementation to measure the cheap one against, and it need not become the default to be useful.
[The concordance's §8](drtmle/theorem-concordance.md#8-cross-fitting-is-not-in-the-sources-and-the-argument-for-it-item-15) holds
the status.

**A1a narrowed where this can be measured, and did not touch the item itself** — but the reason it
gave was wrong, and A1b found that by building the thing the reason was about. This paragraph used
to say the comparison runs at **saturated** reductions "where every conditioning cell is a singleton
and the pooled construction and any nested one return the same arrays". On that law the design takes
three values over a thousand rows, so the cells are *not* singletons; and saturation of the
reduction is not what decides it either, since under a primary learner that learns any reduction
learner returns different arrays. What actually makes that module silent is `cross_fit=False` and
oracle primary learners: one fold has no complement to nest inside, and a learner that ignores its
training rows returns the same function whichever rows it saw. The conclusion survives and the
reason is now the true one — which matters, because the false one would have licensed reading a
*cross-fitted* saturated fit as evidence, and that is the mistake
[stop-ship 14](#stop-ship) exists to prevent. `tests/unit/test_nested_reductions.py` asserts the
corrected statement rather than describing it.

###### What A1b landed

*Closes item 15's reading and supplies the instrument for its measurement.* **Landed.** The entropy
condition closes on the learner; the stability condition is **supported and not shown**, which is
the honest state and not a hedge — see [what the dispatch
measured](drtmle/investigation-log.md#what-the-a1b-dispatch-measured).

`DRTMLE(reduced_crossfit="nested")` — a non-default reference construction in which fold `k`'s
reduced regressions are trained on designs *and targets* from primary models that left fold `k` out
as well, and still predict at the production design.
[The concordance's §8](drtmle/theorem-concordance.md#8-cross-fitting-is-not-in-the-sources-and-the-argument-for-it-item-15)
is the argument, [the validation plan's §7](drtmle/validation-plan.md#7-the-cross-fitting-construction-piece-a1b)
is the frozen rule, and `--reduced-crossfit nested` is the arm that reads it.

**The two tracks turned out to be one track, and that is the piece's finding.** This page kept them
apart — a proof *or* a reference estimator — and the argument, once written, says why they are not
alternatives. Splitting fold `k`'s empirical-process term leaves a residual `(P_n − P_0)Δ_k` that no
cross-fitting lemma reaches, and `Δ_k` is *by definition* the difference between the pooled
construction and a nested one. So the open condition of the proof is the quantity the reference
estimator computes. Had only one been landed it should have been the arm: a proof whose key
condition is unmeasured is item 13's shape one level up.

**Four decisions in it are forks rather than transcription.**

- **Both halves of a reduced regression are generated regressors**, and this page said only the
  designs were. `Q_r`'s target is a residual of `Q̄̂` and `g_{r,2}`'s is a quotient by `ĝ`; only
  `g_{r,1}`'s target is data. A construction replacing the designs alone would have removed half
  the dependence and reported itself as having removed it all.
- **The inner split is the outer split, used twice — leave two folds out.** One `fit_mask` keyword
  of `cross_fit_predictions`, so no second split, no new randomness the `random_state` does not
  determine, and cluster integrity inherited. The inner training set is `(K−2)/K` of the sample
  against production's `(K−1)/K`, and that gap is between two *cross-fitted* models of one nuisance
  — not the in-sample-versus-out-of-sample shift `fit_reduced` rejects per-outer-fold designs for.
- **The fluctuation is carried through the solvers, not reconstructed from `(initial, ε)`.** The
  reconstruction looks free and is wrong exactly where it matters: the outcome tilt is applied once
  per Newton step and shrunk after each, and `solve_bounded_mechanism` clips into `g_bounds`, so a
  net offset recovers the endpoint only on a fit where nothing touched a bound. The fits that touch
  one are the weak-overlap fits this arm is compared on.
- **`targeting="one_step"` is refused by name**, on cost rather than on derivation, and says so.

**The entropy half of the argument is settled by the learner, and the answer is narrow and
pleasant.** Because the reduction is univariate, the condition falls on a one-dimensional class and
not on the primary nuisances at all. `mean`, `glm`, `glmnet`, `gam` and `boost` are inside it —
boosting because this package pins its hyperparameters rather than tuning them, which is load-bearing
rather than incidental. `forest` is not, so **`library="rich"` is outside the guarantee by
declaration**, and `reduced_*_learner` defaults to the primary spec, so a caller reaches it without
any keyword saying so. That is a scope row, not a defect, and it sits beside item 25's.

**The dispatch supports the stability half and does not resolve it**, and the wording is the
finding rather than a hedge. [What it
measured](drtmle/investigation-log.md#what-the-a1b-dispatch-measured): three of §7's four clauses
pass — the `se` ratios, the paired counts against the control, and zero score-check or identity
failures in 144 fits — and **the primary clause passes on `nonlinear` and fails on `linear`**, where
the nested difference is flat across three sizes while the control halves. That is the literal shape
of the falsifier and is reported as such. It is not read as one because on that process the nested
difference sits 3 to 7 times *below* its own control at every size: the falsifier was written for a
construction difference that persists while split noise dies away, and this is a fraction of split
noise throughout. What is stable in all six cells is that the construction moves `ψ` no further than
a redrawn split does, and that where the differences are large enough to have a trend the two shrink
together — `Δ_k` behaving like split noise rather than like a bias.

**This is the second median-based clause at twelve draws to fail to carry a slope**, and it is
filed against the instrument. The restatement §4 had pre-registered as available — read the *ratio*
of the two medians — does not rescue it either. What a further dispatch needs is not more seeds but
the instrument §7 recorded as **not built**: the paired `L₂` distance between the two arms' reduced
arrays, which measures `‖Δ_k‖` rather than its consequence on `ψ`, where the cancellation and the
noise are.

**What it does not close, and neither is claimed for it.** `‖Δ_k‖ = o_p(1)` is supported, not
shown. And `g_{r,2}`'s envelope is `1/lo` with `lo → 0` under `g_bounds="auto"`, so its class is
bounded by a *rate* rather than by a fixed ball; that row pulls against item 25's "bound sequence
eventually below δ", and both are now in the matrix.

**The assumption matrix's cross-fitting row became six, and the count of `unverified` rows went
up.** That is the column working as A1a's revision described: the old row said *nobody has an
argument*, and the new ones say what the argument is, which conditions the learner settles, and
which one a run has to.

##### What the component checklist is, now that it is not a parity run

*This was piece A2, and item 2 is [retired](#closed-since-this-list-opened).* What survives is the
decomposition, and it survives because it was never really about R. **Compare components, not `psi`
and `se`**: several differences cancel at `psi`, and `psi` is precisely the quantity all three
empirical means being zero makes *insensitive* to the corrections. The order to localise a
discrepancy in, the shapes of the laws each component is checked on, and the per-component
tolerances are in [the validation
plan](drtmle/validation-plan.md#3-the-component-checklist-piece-a1a); which test pins which object
is the concordance's `evidence` column.

**A1a finished it, and one line of the instruction turned out to be too narrow.** "Compare
components, not `psi` and `se`" is right about `psi` and wrong to stop at the components: the
last row of the checklist is the *assembled* curve, and nothing that checks only its parts can
see how a fit puts them together. That is what `tests/unit/test_influence_gateaux_drtmle.py` adds
above the theorem-terms comparison, and it is why the two are not one check — the first reaches
the parts at a nonzero `Q_r`, the second reaches the whole from a real fit, and each was watched
to fail against a library mutation the other does not see.

The labels moved with item 21: `reduced_corrections`, the [methodology
section](methodology.md#doubly-robust-inference-what-the-extra-equations-remove) and the guide used
to say **what `drtmle` computes** rather than what the theorem derives, and they now say the two
agree and what the agreement took. **Two claims stay separate**, where there used to be three: that
the algorithm satisfies the theorem, and that it helps in finite samples. The third — that Python
implements the same algorithm as R — is now *provenance* and is not something this repository
checks or will check. Do not let a document slide back into treating it as evidence.

#### B. The loop's exit, and whether what it leaves is what gets reported

**Closes items 11, 12, 19, 20 and item 22's numerical half, and re-measures items 4 and 6.**
Seven things, and they were one piece because one dispatch of `benchmarks/bench_drtmle.py`
produces the evidence for all of them.
Two of the six stopped needing the sweep once their cause was located, and what is left of them is
a convention decision and its tests — which is why this is now three pull requests and not one.

##### B1a — the identity and safety patch

*Opens the closure of items 11 and 20; blocked by nothing.* **Landed.** What it shipped, and
what it deliberately did not, is [at the end of this section](#what-b1a-landed).

The cause of item 20 is located and is not what either of the first two readings supposed. The
execution plan's reading — "at least one recorded score is evaluated at a different state from the
arrays used to build the reported influence curve" — was checked directly and is **false**:
recomputing equation (9)'s score from the returned `fluctuation.mechanism.propensity` and
`fluctuation.reduction.reduced` reproduces the recorded score **bit for bit** on both an uncentred
draw and a centred one. The record is faithful. What differs is downstream of it:

```text
equation (9), as solved     Pn[ H_g · (A − g*) ]        g* RAW,       from solve_mechanism
D*_g, as reported           Qr/ḡ* · (1_a − ḡ*)          ḡ* TRUNCATED, from reduced_corrections
```

Both read the same `g*`. Only one truncates it in the *residual*, and the covariate's denominator
is truncated in both — so the two expressions are **identical on every row the truncation does not
bind and differ on every row it does**, and the reported curve is off by exactly the clipped rows'
contribution. The measurements, the algebra, the four things it settles and a two-row construction
that proves the point with no simulation at all are in [the investigation
log](drtmle/investigation-log.md#item-20-from-discovery-to-cause). Three consequences for this
piece:

- **an immutable state object would not have fixed it.** The plan's `DRTMLEState` is worth having
  and is not the remedy: both expressions already read one state. What is missing from that state
  is the *truncation*, applied twice by two callers;
- **the `1e-11` is real and is measuring the wrong thing.** The loop solved the equation it posed;
  what it did not do is pose the equation whose solution the curve needs;
- **solving one gives no bound on the other.** `Pn[D_{g,b}] − S_raw = Pn[(Q_r/g_b)(g_raw − g_b)]`
  has no reason to be small, and one clipped row of 600 was enough.

So B1a is the patch that makes this class of defect impossible to hide, **without choosing the
convention** — which is what lets it land now. It recomputes each arm's `Pn D*_g`, `Pn D*_Q` and
ordinary score from the exact returned state, exposes the identity residuals `Δ_g(a)` and
`Δ_Q(a)`, adds the exact clipping-bias diagnostic `B_clip(a)`, marks inference **invalid** when
either an identity residual exceeds roundoff or a final correction score exceeds the predeclared
tolerance, and reports the invalidity on the face of the result. The five conditions on how those
identities must be checked — per arm, before the contrast, with weights, on one outcome scale, and
on a fixture where the bound **binds** — are in [the validation
plan](drtmle/validation-plan.md#1-the-invariants-piece-b1a), and the last of them is the
degeneracy [lesson 2](drtmle/investigation-log.md#what-the-sizings-got-wrong) names.

`tests/unit/test_drtmle_fit.py::TestTheReportedCurveIsNotAlwaysCentred` must be **rewritten rather
than deleted**: it currently pins the defect's numbers, and afterwards its fixture is the
regression test that the bounds still bind on that draw.

The immediate next coding task on this whole variant is this and not "make the weak-overlap
fixture pass":

> Make every final score and every correction component a provably identical evaluation of the
> exact returned state, expose the clipping discrepancy explicitly, and invalidate inference
> whenever the final reported score is not negligible.

That patch is valid under every eventual theoretical convention. It gives B1b clean evidence to
decide the final targeting algorithm with -- the decision is that piece's, not the audit's -- and
it prevents B2 or C from producing apparently authoritative results through an internally
inconsistent curve.

###### What B1a landed

`cleverly/validation/drtmle.py` — `correction_check()`, reached as
`res.validation.correction_check()` and folded into `score_check` as two new row kinds. Per draw
and **per arm**, from the exact returned state: each equation's stored score, the mean of the term
the reported curve subtracts, their difference, `B_clip`, and how many rows the bound binds on.

Four decisions in it are worth having written down, because each was a fork:

- **Derived, never stored**, exactly as `score_verdict` is. Nothing is added to
  `ReductionFluctuation` and there is **no format bump** — the records format version 10 already
  serialises are enough, so a fit read back from disk answers with its own arrays. A flag written
  at fit time would be one nothing could check afterwards, on a diagnostic whose whole subject is
  a disagreement between what a fit recorded and what it reports.
- **The curve's own arithmetic, not a second copy of it.** `reduced_corrections` was split into
  `reduced_correction_parts`, which returns `D*_g` and `D*_Q` apart, and the check takes their
  means. An identity checked against a re-derivation of the same formula is not an identity —
  which is not hypothetical: the first version of the test that compared `parts.total()` against
  `reduced_corrections` survived turning the sum into a difference, because by then one called the
  other. `estimators/tmle.py::correction_parts` is likewise module level so that the reported curve
  and the check select the same mechanism.
- **Two failures, worded apart.** An identity residual is a *software defect* and iterating longer
  cannot fix it; a correction score above `tolerance · se/√n` is a *fit that did not solve its
  equations*. `score_check`'s verdict names the first where it applies, because "the score
  equation was not solved" sends a reader to `one_step` and a smaller step size for a fit whose
  solver did its job — which is what happened for two revisions.
- **One scale.** Everything is reported on the outcome's own scale. `Q_r` and the fluctuation's
  residual live on the `[0, 1]` scaled outcome and the reported curve carries `scaler.range`, so a
  correction and `se/√n` are otherwise a factor of `range` apart —
  [lesson 8](drtmle/investigation-log.md#what-the-sizings-got-wrong)'s pattern in a second place.
  Dropping the factor was one of the seven mutations run, and it fails a test.

**The sign in the plan was one orientation out and the plan's is kept.** `B_clip` is defined there
with `g_raw − g_b`, and the residual the check reports is `stored − reported`, so the two are
*negatives* of one another: `Pn[w D*_g] − S_g^stored = Pn[w B_clip]`. It reproduces to floating
point, per arm, which is what makes `B_clip` a check on item 20's diagnosis rather than a new
column.

`IDENTITY_TOLERANCE = 1e-12`, absolute, on the outcome scale and deliberately not relative to the
score: the quantity is a difference between two evaluations of one expression and its right value
is zero. Measured on `nonlinear_dgp`, a holding identity sits at `2e-19` and the smallest real
failure seen at `7e-08`, so the bar has seven orders of headroom below and four above.

**It is an instrument and not a remedy.** The identity still fails on a clipping draw, which is
the state B1b decides. Nothing about a fit's `psi`, `se` or curve moved; what changed is that such
a fit now says which arm, which equation, and that the cause is an expression rather than a
solver.

##### B1b — the theorem-conforming targeting decision

*Closes items 11 and 20.* **Landed.** `cleverly.fluctuation.mechanism.solve_bounded_mechanism`
solves equation (9)'s score at the **truncated** tilt — the expression the reported curve
carries — and the alternation carries that truncated array forward, so the stored score and the
term the curve subtracts are one evaluation of one expression at the returned state. [What it
shipped](#what-b1b-landed) is at the end of this section; the reasoning it was chosen on is
below, and none of it changed on contact with the implementation.

The defect could be removed under **more than one** targeting design, and an earlier
revision of this page said there were exactly two. There are at least four, and the difference
matters because matching `drtmle`'s convention would make the recorded score and the reported
correction refer to one expression and would **not** make hard clipping after a logistic
fluctuation solve that expression's score equation. `Pn[H_g(A − clip(expit(·)))] = 0` is not the
canonical logistic score, and a hard clip is a non-smooth projection applied *after* the
optimisation, so the unconstrained first-order condition is not the first-order condition of the
clipped state. R does not exhibit Python's inconsistency — `fluctuateG` applies
`pred[pred < tolg] <- tolg` and returns *that* as `gnStar`, so it has one array — and that is
internal consistency, not a theorem.

**And the theorem clips nothing at all.** Its mechanism update is a plain `expit` fluctuation with
no projection after it, and it assumes the *true* `g_0` is bounded away from zero rather than
truncating a fitted one. So the theorem as written supports neither the current hybrid nor R's
post-fit clipping as an exact step — which is what makes this a derivation rather than a choice.
The four candidates, their costs, and the four-criterion decision hierarchy that ranks theorem
fidelity and exact final-score validity first are in [the validation
plan](drtmle/validation-plan.md#2-the-targeting-candidates-piece-b1b) — a hierarchy that used to
have R parity at its foot and now does not have it at all.

Two things to carry into the implementation. `solve_mechanism` is shared with `ipsi`, which is a
regression surface: **the change belongs at the `DRTMLE` call sites, not in that function.** And
if a bounded convention is adopted, [limitation 6](#limitations-recorded-rather-than-fixed) gets
*worse* rather than better, since a truncated residual is not the canonical logistic score.

###### What the prototype settled, and it is what the implementation followed

**The piece was sized against a fitted prototype rather than against the candidate table**, and
the first thing that run settled is that the table's axis is not the discriminating one. Two hooks
on `targeting`'s module namespace, nothing in the library moved, three draws and a forced bound,
three conventions; the numbers are in [the investigation
log](drtmle/investigation-log.md#what-the-b1b-prototype-measured) and four readings of them decide
this section.

- **What carries the defect is the array the alternation carries forward.** `targeted_g =
  mechanism.propensity` is the *raw* tilted mechanism and the next round offsets from its `logit`,
  so a row outside the bounds stays outside for the rest of the fit. Carry the **bounded** array
  forward instead and the identity holds at the exit near-automatically, because at a fixed point
  `ε → 0` and the two arrays coincide there. Measured: both candidates exit with **zero** clipped
  rows on draws where today's convention clips 1 and 167, and `Δ_g` falls from `5.8e-04` and
  `3.7e-03` to roundoff.
- **Two of the four candidates cannot satisfy criterion 1 by construction**, which is the paragraph
  the decision hierarchy asks for. **C** is today's solver with the curve made to follow it, and a
  `D_g` whose residual and denominator sit at two different mechanisms is no theorem's `D_g` at
  all, whatever it is the first-order condition of. **B**, the untruncated equation, *is* the
  theorem's own step and stays the definition of the estimator wherever the bound is slack, but a
  fitted `g*` is not bounded away from zero the way the theorem's `g_0` is assumed to be, so it
  cannot be the default. That leaves
  **A** — clip after the logistic solve, `drtmle`'s convention — and **D**, solve the bounded
  equation directly.
- **A and D are one fit wherever the bound does not bind at the fixed point, and separate by four
  orders where it does.** At the `auto` bound they agree on every draw run. Forcing
  `g_bounds=(0.15, 0.85)` on the weak-overlap draw leaves A's final scores at `6.8e-06` and
  `2.1e-06` against an inferential threshold of about `4e-06`, and D's at `2.1e-10` and `8.0e-10`.
  That is criterion 2 doing the work it is ranked second for, and the separation is the predicted
  one rather than a numerical accident: A's substep solves the **pre-clip** score and the clip is
  a projection applied after it, so a fixed point with clipping is a fixed point of neither
  equation.
- **D is a regression surface where nothing clips.** On the module fixture `psi`, `se` and both
  stored scores agree with today's fit to every digit printed — expected rather than lucky, since
  with the clip slack on every row D's equation *is* the logistic score.

**So the choice is D, and the remaining question was which D.** [The concordance's
§7](drtmle/theorem-concordance.md#7-truncation-is-not-in-the-theorems-algorithm) had already said
this without the numbers: if a finite bound is required in practice it wants *a bounded submodel
or a constrained estimating equation whose final score is the theorem-defined score* — **not a
projection applied after an unconstrained optimisation**, which is A. What the prototype adds is
that the difference is `6.8e-06` against `2.1e-10` rather than a matter of principle only. But §7
also says a **smooth** bounded submodel is likely preferable to hard clipping, and the prototype
ran the hard one:

- **D-hard**, which is what landed: `clip` inside the estimating equation. The final score is
  exactly the declared estimator's and the non-smoothness lands on the *solver* rather than on the
  score's validity — but a root can fail to exist, and rows pinned at the bound contribute nothing
  to the Jacobian, which is why the failure path is named rather than inferred.
- **D-smooth**, and it was measured before being rejected: fluctuate inside the bounds — `g_ε =
  lo + (hi − lo)·expit(logit((ĝ − lo)/(hi − lo)) + εH_g)` — so the mechanism cannot leave them and
  no projection is applied at all. `F` is then smooth and criterion 4 is easier to argue. It
  **loses twice**, and the first of the two is the decisive one: it is a *different submodel on
  every fit*, not only on the clipping ones, so at inert bounds of `1e-6` it moved the no-clip
  fixture's `psi` by `2.7e-03` standard errors where D-hard moves it by zero — which would break
  `tests/unit/test_influence_gateaux_drtmle.py`'s `1e-12` window on the module whose whole point
  is that tolerance. And where the bound does bind it left the final score at `1.5e-07` against
  D-hard's `2.1e-10`, because its derivative `(hi − lo)·u(1 − u)` collapses near the bounds.

**The comparison this section promised was run before anything was written**, on the three
fixtures below plus the forced bound, and the [investigation
log](drtmle/investigation-log.md#what-the-b1b-prototype-measured) has both tables. §7's preference
for a smooth submodel is an argument *against a projection applied after an unconstrained
optimisation*, which is candidate A; D-hard puts the clip inside the equation, so the stated
reason does not reach it.

###### What B1b landed

`cleverly.fluctuation.mechanism.solve_bounded_mechanism`, called from the two `DRTMLE` sites in
`solve_with_reduction` and `_close_at_frozen_reductions`. `solve_mechanism` itself does not move —
it is `ipsi`'s, and that is a regression surface.

Four things in it are decisions rather than transcription:

- **The unconstrained solve runs first and is returned untouched when nothing clips.** Not an
  optimisation: where the clip is slack on every row it is the identity, so the unconstrained root
  *is* the bounded root, and a draw whose bound never binds comes back bit for bit — down to
  `hessian_condition` and `loglik`. That is what makes every module fitting at inert bounds a
  surface **by construction** rather than by measurement. The case that makes it load-bearing
  rather than a shortcut is a plain solve that *failed*: without it the bounded branch would go
  looking for the root with a different solver and find it, which is a better answer to a
  different question, and `tests/unit/test_bounded_mechanism.py` pins exactly that.
- **The root finder is `scipy.optimize.root`'s `hybr` and the *verdict* is this package's.** `F`
  is only piecewise smooth, so a step taken with one active set can land in another; a hand-rolled
  damped Newton stalled at `1.9e-04` on a fixture where a root exists at `1e-17`. The solver
  proposes an iterate and the score decides whether it is a solution, so there is one definition
  of converged here rather than two — which is the reason `mechanism.py` gives for sharing
  `_newton_logistic`, applied where it still can be.
- **A root need not exist, and that is never silent.** Pin every row and the clip is flat
  everywhere, so no `ε` moves `F` at all. Such a fit reports `failure = "bounds_pinned"`, which is
  an existing `TargetingFailure` whose wording already says exactly this, and `score_check`
  surfaces it. The trap the tests pin is the *near* case: at a bound one notch tighter than the
  fixture's the score is a small `1e-05` and there is still no root, so a threshold on the score
  alone would call it converged.
- **The carried array is the truncated one**, in the alternation and the closing pass, so every
  later step — the reduction refits, equation (8)'s covariate, the final regression and
  `correction_parts` — reads one mechanism. This is the half the prototype found to be
  load-bearing, and reverting it alone puts the identity back at `1.6e-06`.

**Measured on the four fixtures the defect was characterised on**, against a threshold of
`4e-06` to `6e-06`: every state identity holds at `1e-17` or better, every final correction score
is `1e-09` to `1e-10`, and all four pass their score check — including `weak_overlap` seed 0 at
both the `auto` bound and a forced `g_bounds=(0.15, 0.85)`, where 167 and 375 rows clipped. `psi`
moves by `0` on the no-clip fixture, `0.003·se` on the one-clipped-row draw, `0.06·se` on
`weak_overlap` at `auto` and `0.69·se` at the forced bound — the last two being exactly the case
the acceptance criterion below describes, a different mechanism carried into every later step.

**Two things it does not close, and neither is claimed for it.**
[Limitation 5](#limitations-recorded-rather-than-fixed) — equation (9)'s covariate reads the very
mechanism it tilts, so one solve leaves a residual at the post-tilt covariate — is untouched, and
measured to be: the final scores sit at the same `1e-09` to `1e-10` they did before, because the
outer loop is still what makes the direction self-consistent.
[Limitation 6](#limitations-recorded-rather-than-fixed) was priced above as getting *worse* under
a bounded convention. On these four fixtures it does not: the closing pass's mechanism stage binds
on its cap on all of them, exactly as it bound on 94 of 96 before. Four fits were not 96, and
[B2b](#b2b--the-dispatch-and-what-it-decides) has since re-measured: **96 of 96**, so the
prediction is confirmed and the two fits that used to stop otherwise no longer do. It is the
smallest confirmation available — a cap that nearly always bound now always binds — and it is the
one place a bounded convention cost anything.

**One of B1a's five conditions had to be replaced, and this is the only place that noticed.** Its
fifth is that an identity be checked *on a fixture where the bound binds*, witnessed by
`CorrectionRow.clipped`. That witness is now **vacuous**: a converged bounded tilt lies inside the
bounds, so `clipped` is 0 on every fit, including the draws where 5 and 375 rows clipped before. A
test selecting its fixture on `clipped > 0` would select the empty set, which is
[stop-ship 14](#stop-ship)'s shape — a check agreeing where it could not have disagreed — in a
second place.

**The replacement is not the one this page proposed, and the correction is worth keeping.** The
plan said the witness should be the *initial* mechanism's clipped share, on the reasoning that it
is a property of the draw. It is a property of the draw and it is **zero on the draw item 20 was
found on**: nothing about that fit's initial mechanism leaves the bounds, and what clipped was the
*tilt*. The witness that works is `CorrectionRow.margin`, how close the targeted mechanism comes
to either bound as a fraction of the interval — `1.2e-06` on that draw against `0.14` on its
sibling, because a constrained root sits *against* the boundary of the feasible set. It is not a
proof that the constraint was active, and nothing derivable from the returned arrays is, since the
trajectory is not on the record; what it is is a property that separates the two draws by five
orders and that the fix cannot manufacture.

**And it moves what [B2](#b2--the-sweep-on-the-corrected-implementation) should expect.** The
`weak_overlap` seed-0 draw used to fail its score check with a verdict saying the standard errors
do not describe the estimate, and now passes with its worst final score at `6.6e-10`. One draw is
not the 24 that motivated a weak-overlap refusal and B2 re-measures all of them — but the standing
instruction not to predeclare that refusal now has a measurement behind it rather than only a
caution.

Two acceptance criteria were stated wrongly in the previous revision and are corrected in the
validation plan; both corrections came from the third review and both are right.

- **A weak-overlap fit is not required to pass.** "A `weak_overlap_dgp` fit whose score check now
  passes" was a B1 deliverable, and it conflates a software identity with a statistical verdict
  while prejudging B2. The criterion is that on the pinned weak-overlap fixture the stored score,
  the recomputed correction score and the influence-curve mean **agree exactly** under the
  selected convention; whether the score then clears the validity threshold is B2's outcome.
- **`psi` invariance is a tolerance, not an axiom.** It holds if B1b changes only how the
  correction is *evaluated* after targeting. It does not hold if the convention changes the
  mechanism carried into later outcome targeting steps, later reduced-regression refits, the
  closing pass or the final targeted regression — and then the plug-in moves legitimately. So
  predeclare `|psi_new − psi_old| ≤ c·se_old`, compare each candidate against the estimate its own
  exact targeted state produces rather than against today's `psi`, and treat a material movement
  as something to investigate rather than as an automatic rejection. **The prototype says to expect
  one**, and where: `ate` moves by `0.02·se` on the ordinary clipping draw and by `0.5·se` on the
  weak-overlap draw at a forced bound. Both are exactly the case the paragraph above describes —
  the mechanism carried into every later step is a different array — so `c` has to be set with that
  in mind, and a movement of that size is a fact about which state was targeted rather than
  evidence against the candidate.

##### B2 — the sweep, on the corrected implementation

*Closes items 12, 19 and item 22's numerical half, re-measures items 4
and 6, and decides the weak-overlap product policy.* One dispatch of `benchmarks/bench_drtmle.py`
produces the evidence for all of it, and it must run **after** B1a at the least: every conclusion
it could draw today is read through a curve that a share of fits have wrong.

**This piece split in two, and the reason is B1's rather than A1's.** B1 split because one half
preceded the other and depended on it; so does this. "One dispatch" was written as though the
script already recorded what [the validation plan's §4](drtmle/validation-plan.md#4-the-sweep-piece-b2)
asks it to record, and it recorded three columns of the eleven — and item 22's numerical half asks
for a comparison against an update order that **did not exist in this repository at all**. So the
instrument is [B2a](#b2a--the-sweep-instrument), which has landed, and the dispatch and its reading
are [B2b](#b2b--the-dispatch-and-what-it-decides). Nothing about the questions changed; what
changed is that they are now answerable by a run.

**Item 22's numerical half arrived here from A1a**, and the move is the grouping rule rather than
a demotion. The theoretical half is closed — the paper prescribes a fixed point, not a route — and
what is left is whether the paper's order and this one land on the same one *in practice*. That is
a second alternation run over a sweep of draws: it shares this piece's dispatch and shares nothing
with a document audit. Add it as a column here, not as a study of its own, and read
[A1a](#a1a--the-theoretical-audit)'s two cautions with it — both orders run in this repository,
and **no comparison of fluctuation coefficients across algorithms**.

###### B2a — the sweep instrument

*Lands the columns, the second update order and the comparison arms; closes nothing on its own.*
**Landed.** It is the half of B2 that had to precede the dispatch, and this section is what it
shipped.

**`DRTMLE(update_order="paper")`** is [the concordance's §6](drtmle/theorem-concordance.md#6-the-recursive-algorithm-item-22)
recursion implemented beside this package's: equation (8), then `g_{r,1}` and `g_{r,2}` at the
**once-updated** outcome regression, then equation (10), then `Q_r` at the **twice-updated** one,
then equation (9). Four things about it are decisions rather than transcription.

- **It shares the stopping rule, the stall test and the closing pass**, because the question is the
  *route*. An arm of a comparison that also carried its own convergence criterion would answer
  nothing about either.
- **The two orders are one function with a branch**, not two implementations. `ReductionSpec.order`
  carries the declaration, exactly as `guard` does, so `TMLE._solve_reduction` stays free of a
  setting one subclass has and nothing is duplicated that could drift.
- **Equation (8)'s score is restated before the exit test**, unconditionally. Under the paper's
  order it is solved first and steps 4 and 6 then move both the regression it fluctuated and the
  mechanism it divides by, so the loop would otherwise stop on a number for a state that is gone;
  under this package's it is solved last and the restatement is a **bit-for-bit no-op**, which was
  measured rather than assumed — `solve_fluctuation` computes the score it returns *after* its
  loop, by that same expression at the iterate it returns.

  It shipped conditional on the order and is now unconditional, and the reason is
  [lesson 12](drtmle/investigation-log.md#what-the-sizings-got-wrong). **Deleting the restatement
  was run and 68 of the module's 69 tests still passed** — the closing pass re-solves all three
  equations and makes the *reported* fit identical either way, which is item 12's shape in a second
  place. A branch that only one order exercises is exactly what that lesson says to remove rather
  than to guard, so there is now one call, the invariant it rests on is pinned one level down in
  `tests/unit/test_fluctuation_score.py`, and the call-site test carries a single expectation
  instead of one per order.
- **The default path is a regression surface and was measured as one**: 111 tests across
  `test_drtmle_fit.py`, `test_influence_drtmle.py` and `test_bounded_mechanism.py` pass unchanged,
  and the restatement is called from the one branch rather than from both.

**The sweep now prints three tables** where it printed one — the exits as before, *Where weak
overlap enters* (the clipped-row share, the margin, the smallest `g`, the per-arm effective `n`,
and the 99th percentiles of all three clever covariates and of `Q_r`, `g_{r,1}` and `g_{r,2}`), and
*What the reported curve rests on* (B1a's identity residual and `B_clip`, the standardised score
`|P_n S_j|/sd̂(S_j)` reported **beside** the stopping rule rather than folded into it, the share of
the worst score carried by its top 1%, 5% and 10% of rows, and the Hessian conditioning). Two
column definitions had to change on contact with the code and [the validation
plan](drtmle/validation-plan.md#4-the-sweep-piece-b2) carries both: the clipped-row share is read
at the **initial** mechanism, since a converged bounded tilt clips nothing and a column read at the
exit would be zero on every row — stop-ship 14's shape again; and equation (9) has no Hessian to
report at all, because B1b made that solve a root find rather than a Newton step.

**And one instruction in the plan could not be executed as written.** The oracle-reduction run was
sized as costing nothing "because the datasets already know their truth". They know `Q̄₀` and `g₀`;
they do not know the reductions, which are conditional expectations given **fitted** objects and so
have no truth the process can supply. `--reduced-learner` is the substitute on the continuous
processes — vary the reduction's learner and see whether the failures move — and it is labelled as
that rather than as an oracle; what a genuine one would take there is **item 24**.

**Where the oracle does exist it has since been built, and it answers more than the sweep asked.**
On the exact law the conditional expectations are finite sums, so
`tests/unit/test_oracle_reductions.py` injects them through `ReductionSpec.refit` — recomputed at
the current targeted pair every round — and runs a real alternation on them. What it found is that
**with the reductions exactly right the fit recovers the truth while both primary nuisances are
wrong on purpose**, to `3.6e-08` and to `1e-12` where no mechanism equation is solved; that the
saturated learner reproduces the oracle to `1e-14` over a whole alternation, which is the control;
and that **a wrong reduction moves `psi` by 0.36 to 0.80 of a standard error while leaving every
score solved.** The last of those inverts the question this arm was written to answer: a fit whose
*scores* fail is not a fit whose reductions were noisy, because a bad reduction does not show there
at all — it damages the estimate silently, which is the case an interval cannot see.

**What the arms cost, since that decides what B2b dispatches.** Each refits every draw, so each
roughly doubles the run; `--order paper` more than doubles it, having taken 22 rounds against 8 on
the first draw the two were compared on. All three are off by default and are workflow inputs.

**One number came out of the smoke runs that B2b should expect to have to explain.** On a 400-row
`nonlinear` draw the two routes' `ate` differed by **0.22 of a standard error**, with both fits
passing their score and identity checks; on the 600-row module fixture the same difference was
`9e-03` of one and the *standard errors* differed by 2.3%. Neither is a defect and both are
consistent with the theorem — step 7 constrains three empirical means, and two states satisfying
them can differ — but "the two routes agree" is not what one draw showed, and the sweep should
report `|Δψ|/se` **by size**: if both routes are asymptotically linear with the same curve, that
number has to shrink with `n`, and it is a claim with a direction rather than a reassurance.

**That number had no yardstick, and the remediation gave it one.** A route difference and a *fold
split* difference are the same number until something separates them, so the sweep gained a
`reseed` arm — same estimator, same data, one different fold seed — paired the way the paper arm is,
and a table that reads the two together. At smoke scale, **two draws and therefore not a finding**,
the medians are `9.97e-02` for the route against `8.23e-02` for the reseed. On present evidence the
0.22 is what a refit does rather than what the route does, which is the opposite of what this
paragraph was written expecting. The rule that decides it, written before the dispatch that will
judge it, is [in the validation
plan](drtmle/validation-plan.md#the-update-order-rule-frozen-before-the-dispatch), and it hangs on
a **count** rather than an interval because twelve draws do not support a Monte Carlo interval on a
median and this package has no estimator for one.

###### B2b — the dispatch, and what it decides

*Closes items 12 and 19, re-measures items 4 and 6, decides the weak-overlap product policy, and
leaves item 22's numerical half answered on one process of two.* **Landed.** [What it
measured](drtmle/investigation-log.md#what-the-b2b-dispatch-measured) is in the investigation log
beside the first sweep's, which stays as the *before*: it measured a criterion that has since
changed and an implementation that has since been fixed, so it is a record rather than a baseline
to reproduce.

**Five questions were put to it and four came back settled.** Each is below with the number that
settled it, and the fifth is the one to read carefully.

- **the exit distribution under the current rule** — it *inverted*. `tol/stall/cap` was 2 / 86 / 8
  and is **87 / 8 / 1**, at a median of 4 to 9 rounds against 12 to 24. Nothing about the iteration
  changed; the exit test reads a different ruler. [Limitation 4](#limitations-recorded-rather-than-fixed)
  is rewritten around it, and the incidental consequence is that the whole sweep costs a **seventh**
  of what it did — 378s against 2,588s — which makes every runtime estimate on this page and in the
  validation plan stale in the same direction, [piece C](#c-the-demonstration)'s included;
- **whether the closing cap still binds on 94 of 96** — it binds on **96 of 96** on that grid, so
  [limitation 6](#limitations-recorded-rather-than-fixed)'s standing prediction that a bounded
  convention would make it worse is confirmed, in the smallest way it could be. It had been carried
  as a guess for two revisions. Adding `n = 2,400` shows it is *not* universal — 102 of 108 on
  `nonlinear`, every exception at the largest size — which nothing had looked for and which makes
  it a limitation that weakens with `n`;
- **whether `weak-overlap`'s 23-of-24 score-check failures survive B1b** — they do not. **0 of 24**,
  and 0 of 36 again at three sizes, with the worst standardised score down from `1.1e+00` to
  `2.1e-07` on draws whose overlap columns are unchanged. That is what takes the product decision
  above;
- **the overlap columns that say which of the five places a failure came from** — with no failures
  left to attribute they instead say what did *not* change, which is the load-bearing half: a third
  of `(row, arm)` pairs still clip, `min g` and `min gr1` still round to zero, `ess/n` is 8–13%.
  One column still separates the process from every other and is recorded rather than acted on:
  `q99 h(10)` at `2.49`, an order above the rest and rising with `n` where they fall;
- **`|Δψ|/se` between the update orders, whether it shrinks with `n`, and how it compares with what
  a different fold split moves** — this is item 22's numerical half and it is **not** closed. The
  [rule frozen before the
  dispatch](drtmle/validation-plan.md#the-update-order-rule-frozen-before-the-dispatch) is a
  conjunction over both processes and its first clause fails on `weak-overlap`. Neither is the
  falsifier met: the *reseed* difference fails to shrink there too, so the yardstick is as noisy as
  the thing measured. On `nonlinear` every clause points one way and the route difference shrinks
  by 7.8 over a fourfold `n` while a refit's shrinks by 2. [The
  reading](drtmle/investigation-log.md#the-two-update-orders-against-the-yardstick-of-a-fold-split)
  keeps both, and the sharpener the rule itself names — more seeds — is what was dispatched next.

**It was four dispatches, not the two this section planned.** The main sweep is four processes at
two sizes with the arms off. The update-order question is its own — two processes, **three** sizes,
the paper arm and the reseed control — because a rate needs three sizes; it ran as one dispatch per
process against a runner cap that turned out to have thirty times the headroom it was thought to,
and then again at three times the seeds because twelve did not resolve `weak-overlap`. Both
configurations, the rule, and what would falsify it are [in the validation
plan](drtmle/validation-plan.md#the-update-order-rule-frozen-before-the-dispatch).

**24. An oracle reduction on the continuous processes needs the fitted learners at rows the fit
never saw — and the companion is where it keeps them.** A reduced regression conditions on `ĝ(a|W)`
or `Q̄̂(a, W)` — *fitted* objects, and univariate: one column each — so its truth is not something a
DGP can supply, and evaluating it means predicting those nuisances away from the fitting sample.

**The original form of this item said "nothing here keeps them", and that has been false since
C2.** `cross_fit_predictions` does discard every per-fold model, and `NuisanceEstimates` does carry
arrays only — both deliberately, so that everything reached through `retarget` targets what the fit
declared without a learner being refitted. But `CompanionEstimates` holds `outcome` and `propensity`
**one copy per outer fold, evaluated at the companion rows by the same fitted model objects**,
filled in by `cross_fit_companion` at fit time; and `ReductionSpec.refit` is handed those copies at
the *current targeted* pair on every round of the alternation (`targeting._reduction_inputs`). So
the construction is available without a source change: fit the reference on the companion, where
the design values and `P₀`'s own weights `g_0(a|W)/points` are both known, and predict at the
production design values the fit already carries.

**What is left is a declaration constraint, not an absent capability**, and it is a virtue rather
than a cost: the population grid has to be fixed *before* the fit, via `evaluation=`, which is what
makes the reference conditionally fixed given the fit — the thing A1b's stability argument wants of
anything entering the reductions.

Two things stop the remainder being a gap in the evidence. On the **exact law** the oracle exists
and is built (above), and it is where the question has an answer rather than an approximation. On
the continuous processes `--reduced-learner` measures the same *effect* — whether a different
reduction moves the fit — without claiming to be a truth. What is still genuinely unavailable is
the magnitude of a continuous-process reduction's error **at an unvalidated reference**: a
reference is a numerical estimate of a conditional expectation and carries a smoothing bias, so
until its own fidelity is gated against something that is not its own refinement, no number here
should be read as that magnitude. **[E2](#e-what-c3c-handed-back) is that gate**, and this row
moves when it lands rather than when the instrument does.

**The diagnosis stays widened even though the cause is found.** `1/g` in equation (8) is one of
*five* places weak overlap enters, and B1 accounts for the score failure without saying the other
four are harmless. The five places, the columns the sweep must add — the clipped-row share first
among them — the run that separates a noisy reduction from a wrong equation, and
the truncation-curve caveat are in [the validation
plan](drtmle/validation-plan.md#4-the-sweep-piece-b2).

*The first sweep measured the criterion that was replaced, and the rerun has happened.* [That
table](drtmle/investigation-log.md#how-the-alternation-exits) is the evidence item 7's change was
argued from, which is the right way round — the failure had to be characterised before the
threshold moved — and it left the exit distribution under the current rule uncharacterised. The
rerun was one dispatch and **six minutes**, not the 45 this paragraph budgeted, and it says
`tolerance` is now the norm at scale rather than on the six fits looked at: **87 of 96**, against
2. It re-measured items 4 and 6 for free, as predicted.

*The absolute bar is a proxy for the one it cites.* `score_check` compares against
`DEFAULT_TOLERANCE * se / sqrt(n)` using the fit's actual `se`; `targeting._solved` substitutes
`_NEGLIGIBLE / n`, which assumes `se = O(n^-1/2)` on the scaled outcome rather than measuring it.
It is conservative exactly where it matters — under weak overlap `se` is large, so the loop's bar
is the stricter one — but "conservative on the cases we looked at" is not "correct", and a fit
with a very small `se` is the untested direction. Passing the realised `se` in would remove the
assumption; it was not done because the loop runs before the estimate exists. **The way out of
that circularity is better than passing `se` in: the loop's bar should not be a proxy for the
reported one at all.** Asymptotic linearity asks for `P_n D = o_p(n^(−1/2))`, and the honest
finite-sample rendering of `o` is a deterministic sequence `c_n/√n` with `c_n → 0` slowly — a
*numerical* criterion, stated as one, with the standardised score `|P_n S_j| / sd-hat(S_j)`
reported afterwards as a separate diagnostic rather than folded into the stopping rule. That
separates the two things `_NEGLIGIBLE / n` conflates: when to stop iterating, and whether the fit
that came out is entitled to a Wald interval. The second is `score_check`'s job and
[item 16](#closed-since-this-list-opened)'s.

**The product decision belonged to this piece and is taken: `DRTMLE` does not refuse under weak
overlap, and no diagnostic is predeclared for it.** The proposal was that if the sweep still found
no stable region the estimator should refuse or invalidate on a predeclared diagnostic rather than
warn — a warning being easy to miss on a method whose only purpose is inference. The whole
evidence for it was 23 of 24 failed score checks on `weak_overlap_dgp`, and
[B2b](#b2b--the-dispatch-and-what-it-decides) re-measured that on the same seeds: **0 of 24**, and
0 of 36 again at three sizes on the order dispatch, with the worst standardised score falling from
`1.1e+00` to `2.1e-07`. The draws did not get easier — a third of their `(row, arm)` pairs still
clip at the initial mechanism, `min g` and `min gr1` both round to zero, and the per-arm effective
`n` is 8–13% against 41–47% elsewhere — so the failure was
[B1b](#b1b--the-theorem-conforming-targeting-decision)'s convention mismatch and refusing on
overlap would be refusing on the symptom's former proxy. **A refusal has to be argued from evidence
that exists**, and after B1b none does.

Three things stand in its place rather than nothing:

- **the ordinary positivity warning**, which fires on these fits already (29% of units outside the
  bounds on the seed-0 draw) and is the honest signal that the *design* is thin;
- **`score_check` on the face of the fit**, [item 16](#closed-since-this-list-opened), which is the
  per-fit answer a predeclared population-level threshold would have been a poor substitute for.
  It is not vacuous here: one *control* fit — a different fold split of a `weak-overlap` draw whose
  base fit passes — does fail, at a rate near 1 in 100 rather than 23 in 24. Such a fit says so;
- **two columns that still separate `weak-overlap` from everything else, recorded rather than
  acted on.** `q99 h(10)` reaches `2.49` against `0.06`–`0.23`, and the largest 1% of rows carry
  28–37% of the worst score's absolute mass against 7–12%. Neither costs a fit its score check
  today. A score driven to `2e-07` by a handful of large rows cancelling is not the same object as
  one that is small rowwise, and if a predeclared diagnostic is ever wanted here, the concentration
  share is the candidate with a reason behind it — not the clipped-row share, which is a property
  of the draw that B1b made harmless.

**19. The alternation's convergence argument proved less than it was read as proving —
[closed](#closed-since-this-list-opened) by B2b.** The diagnosis stays here because it is the
reasoning the wording was changed on. `solve_with_reduction`'s docstring argues that equation (9)
is a weighted logistic MLE of `A | W` and equations (8) and (10) are the outcome quasi-likelihood —
separate factors of the likelihood of `(A, Y) | W` — so each step maximises its own factor with the
others held fixed and "the joint value never decreases". The first review reads the mid-loop refit
of the reductions as breaking that, and it does not: the reductions enter as the *directions* of
the submodels, not as values of the objective, so refitting them changes the next step's direction
and leaves the current joint value where it is, and monotonicity survives. What does not survive is
what the argument is used for. A bounded monotone sequence converges *in value*; that is why the
loop terminates, and it is not why the iterates approach a common zero of three score equations —
under a direction that changes each round, the fixed point of the ascent need not be a stationary
point of anything. The sweep showed the gap in numbers: **86 of 96 fits stalled** at a point the
objective would not climb from, against 2 that reached the tolerance.

**The fix was the wording and the numbers behind it have since moved, which is worth keeping
straight.** The docstring now states the loop as an estimating-equation iteration with empirical
convergence diagnostics, keeps the monotonicity claim for what it does buy — termination, and the
reason not to restart from `Q̄⁰` — and names a stall as an ordinary exit rather than a numerical
disappointment. **That last sentence was written against 86 of 96 and the count is now 8**; the
correction it makes is unaffected, because what was wrong was reading a stall as failure and not
how often one happened.

**20. The reported curve was not centred wherever the mechanism truncation binds, and the
fluctuation rows said it was — [closed](#closed-since-this-list-opened) by B1b.** Kept here in
full, because the whole of this section's diagnosis is what B1b was chosen against and a closed
item that deletes its own evidence is one the next reader has to rediscover. Found by checking
item 18 and not by looking for it. Over 24 draws —
twelve `repeats=2` fits on `nonlinear_dgp` at `n=600` with `glm` on both nuisances, `n_folds=5`,
`learner_folds=3` — **six** leave `Pn[D*_Q + D*_g]` above `1e-8`, at magnitudes from `2e-05` to
`7e-04` on the scaled outcome, every one exiting on `"tolerance"` with no failure recorded and no
ill-conditioned round. On one such draw equation (9)'s **recorded** score is `3.7e-11` while the
mean of the `D*_g` the curve actually subtracts is `-2.3e-04`. The cause is above; the record is
in [the investigation log](drtmle/investigation-log.md#item-20-from-discovery-to-cause).

This is not a `repeats=` defect and refusing that keyword would misdiagnose it: a draw of a
repeated fit is an ordinary fit, and the affected draws include first draws. It is also not a
`nonlinear_dgp` defect — that process is where it was seen because that is the module's fixture,
and the quarter-of-splits rate is the rate at which an ordinary `auto` bound binds on 600 rows.

**Which is why the finding cost one fit and not a cross-language fixture**, and that is worth
recording against the instinct this page had. Both earlier revisions put item 20 in the parity
piece, reasoning
that a divergence between two arrays that should be equal is what a component-by-component
comparison locates. The reasoning was sound and the premise was wrong: there was no divergence
between two arrays. What located it was **recomputing the recorded score from the returned state
in the same process** — thirty lines, one fit, no R — and then finding the recomputation *agreed*
with the record, which is what pointed at the expression rather than at the state. The general
lesson is [lesson 8](drtmle/investigation-log.md#what-the-sizings-got-wrong); the specific one is
part of why that piece is now [retired](#closed-since-this-list-opened).

`score_check` **did** catch it, on the *influence-curve* rows, which are computed from the curve
rather than from what the solver recorded — so a fit in this state said so on its own report
rather than printing an interval like any other. That is item 16 arriving on the first case nobody
constructed for it, and it is the only reason this was seen. From
[B1a](#b1a--the-identity-and-safety-patch) it was caught *as itself* as well: per arm, per
equation, against the score the loop stored, with `B_clip` reproducing the discrepancy to floating
point and the verdict saying **defect** rather than **did not converge**.

**[B1b](#b1b--the-theorem-conforming-targeting-decision) closed it**, and items 11 and 20 close
together because they were always one failure — the loud version under weak overlap and the quiet
one on a quarter of ordinary splits. Equation (9) is now solved at the truncated tilt the curve
reads, and the alternation carries that array forward. What makes the closure a measurement rather
than an argument is that B1a's rows are unchanged: the same `1e-12` bar, the same per-arm
recomputation, the verdicts the other way up. The three fixtures this section's numbers came from
report identities at `1e-17` or better and final scores at `1e-10`, and so does a
`weak_overlap` fit at a forced `g_bounds=(0.15, 0.85)` where 375 rows clipped.

Item **23**, found by the same instrument on the same run, was in this section and is
[closed](#closed-since-this-list-opened).

#### The supported contract, and item 25

**This is not a piece and it is not research. It is the sentence every number C produces is read
under, and it has to be frozen before C rather than inferred from C.** Which options a fit may use
is already enforced in code and listed in three places — the refusals in `estimators/drtmle.py`'s
docstring, `tests/unit/test_drtmle_fit.py::TestTheRefusals`, and [the concordance's assumption
matrix](drtmle/theorem-concordance.md#15-assumptions-and-which-the-implementation-meets). What was
nowhere is the *other* column: of the options a fit may use, which ones the **theorem-backed**
guarantee is claimed for. A supported keyword and a keyword inside the guarantee are two different
statements, and this page had been running them together.

| option | status of the guarantee | what the status rests on |
| --- | --- | --- |
| binary treatment, `reduction="univariate"`, `guard` any subset of the three | **inside** | Theorem 1 as stated, plus the object concordance's `evidence` column and [A1a](#a1a--the-theoretical-audit)'s two anchors |
| **no truncation active** — `ĝ` interior at the initial fit and at the exit, `g_{r,1}` interior | **inside**, and it is item 25's condition | `solve_bounded_mechanism` returns the unconstrained solve *bit for bit* where nothing clips (`test_bounded_mechanism.py::TestTheFastPathIsTheOldSolver`), so on such a fit the estimator **is** the theorem's |
| **a truncation active** | **outside**, and reported as empirically supported rather than theorem-backed | item 25 below |
| `weights=`, fixed | **inside**, by a transport argument written down and checked | item 17: the reductions are `P_w`-conditional expectations because they are fitted by weighted loss, and the mechanism they condition on and divide by is the `P_w` one; `test_remainder_drtmle.py` runs the whole expansion at two tilted laws and keeps the wrong transport as a failing control |
| `weights=`, estimated (`weights_estimated=`) | **refused**, by name | nothing read here says what the reduced regressions of a random tilt are |
| `repeats=` | **inside**, and it needs no DRTMLE-specific derivation | every row is out of fold in every draw, so each draw's fit is asymptotically linear with the same `D` and `mean_r ψ_r` is asymptotically linear with `mean_r IC_r → D`. That is the package-wide argument in [the guide](user-guide.md#cross-fitting-and-cv-tmle), composed with the theorem rather than replacing it; item 18 is the arithmetic under it |
| pooled cross-fitting of the reductions, at `library` in `glm`/`fast`/`default` | **inside**, under a stated entropy condition on the reduction learner **and** a stability condition that is supported rather than shown | [A1b](#a1b--the-cross-fitting-construction)'s argument: the reduction is univariate, so the entropy condition falls on a one-dimensional class whatever the primary nuisances did, and those libraries' one-dimensional fits lie in a fixed bounded-variation ball or a fixed-dimension sieve — [the concordance's §8](drtmle/theorem-concordance.md#8-cross-fitting-is-not-in-the-sources-and-the-argument-for-it-item-15). Its stability half, `‖Δ_k‖ = o_p(1)`, is what `reduced_crossfit="nested"` measures: [the A1b dispatch](drtmle/investigation-log.md#what-the-a1b-dispatch-measured) puts the construction difference at or below a redrawn split's in all six cells, which supports it without closing it |
| the same at `library="rich"`, or any saturated or nearest-neighbour reduction learner | **outside**, by declaration | `forest`'s one-dimensional fits have `O(n)` pieces, so the entropy condition fails. Not refused, because the estimator still computes something and a caller may want it; scoped, exactly as the truncation rows are |
| `att`/`atc`, the other four axes, `delta=`, `intermediate=`, `targeting_scheme="fold"`, `cv_evaluation=True`, `K > 2` arms, `reduction="bivariate"`, composition with `CTMLE` | **refused**, by name | limitation 10 and [piece D](#d-widen-the-scope-to-what-the-sources-derive) |

**Two rows of that table are a correction to a review's reading and are worth saying so.** A
review put `weights=` and `repeats=` together as options where "exact-law and arithmetic tests
establish what the code computes" while the theorem does not cover them, and proposed deriving
them or excluding them from the contract. Neither needs either. `weights=` has a *transport*
argument — not an arithmetic one — and the check that carries it is the whole expansion at two
tilted laws with the wrong transport kept as a control that fails; that is item 17, and what it
already declares as out of scope is the estimated weight. `repeats=` needs no new derivation at
all: it varies the primary split and nothing else, and the averaging argument that makes
`mean_r IC_r` the curve of `mean_r ψ_r` is the one every `TMLE` fit in this package already rests
on. What both rows were missing is the *label*, which is this table. Excluding them would have
narrowed a contract on the strength of a source's silence rather than of a gap in an argument.

**25. The guarantee is claimed for a fit whose truncations are inactive, and that had never been
said.** Truncation is not in the theorem's algorithm — [the concordance's
§7](drtmle/theorem-concordance.md#7-truncation-is-not-in-the-theorems-algorithm) states it as a
finding: there is one mechanism, produced by an unconstrained `expit` fluctuation, and boundedness
is an assumption about `g_0` rather than an operation on `ĝ`. B1b then chose the constrained
estimating equation, which is the right rendering and closed items 11 and 20 — *and* leaves the
implemented estimator solving a different equation from the theorem's on exactly the rows a bound
binds. The three options a review put are the right three: derive the expansion for the constrained
mechanism, restrict the guarantee to fits where the bound is asymptotically inactive, or treat
bound-active fits as empirically supported and outside the theorem. **The second and third are
taken together, and the reason is that the condition is checkable rather than merely assumable.**

- **Where no row clips, there is nothing to derive.** `solve_bounded_mechanism` tries the
  unconstrained solve first and returns it untouched when the tilt stays interior, which is not an
  optimisation but the guarantee that the fit *is* the theorem's estimator — bit for bit, down to
  `hessian_condition` and `loglik`. So on such a fit Theorem 1 applies as written, with no
  rendering to defend.
- **It is the ordinary case at the sizes this variant runs at for the *initial* mechanism, and
  that is measured; it is not the ordinary case at the exit, and this page read it as though it
  were.** [B2b's dispatch](drtmle/investigation-log.md#where-weak-overlap-enters-now-that-it-does-not-fail)
  reports a `clip share` of `0.000` on `linear`, `nonlinear` **and `off-diagonal`** at `n = 600` and
  `n = 1,200` — that is the share of `(row, arm)` pairs outside the bounds at the *initial*
  mechanism — with `margin`, the distance of the targeted mechanism from the nearer bound, at
  `0.11` to `0.20`. `off-diagonal` is the cell shape [C](#c-the-demonstration) turns on.

  **Those margins are medians over twelve draws, and a minority at exactly zero is invisible to a
  median.** [C1's witness](drtmle/investigation-log.md#what-c1s-witness-measured-on-its-first-run)
  read the label per draw on its first run and found **one to two of six** well-overlapped draws
  exiting bound-active, on `linear_dgp` — the easiest process here, chosen for its overlap — with
  the initial `clip share` at `0.000` and `g_{r,1}` well interior in every one of them. So it is
  the *exit* margin alone, and the cause is not positivity: equation (9)'s covariate is `Q_r/g*`
  and `Q_r` **vanishes where the outcome regression is right**, so on a draw whose q99 `|Q_r/g*|`
  is `7e-03` the score's root is an `epsilon` of **24** on the logit scale, which drives rows to
  both bounds while the initial mechanism never leaves `[0.35, 0.86]`. That is item 4 with the
  nuisances swapped — the half of it nothing had written down — and its consequence is that a
  study's cells are **mixed** rather than uniformly inside the contract, so gate 1's clause 0 is a
  *share* per cell rather than a label. The instrument is a scope column and not a verdict:
  every one of those fits passes its score check and every identity holds at roundoff.
- **`weak-overlap` is the other regime and it is not marginal**: `clip share` `0.338` / `0.288` /
  `0.231` at the three sizes, `margin` **exactly** `0.0e+00` at both of the first two — a
  constrained root sits *against* the boundary — and `min gr1` `0.000`. Those fits are the third
  option's population: every identity holds, every score is negligible, the score check passes on
  0 of 24 failures where it used to fail 23 — and none of that is Theorem 1 applying to them.
- **The asymptotic half of the restriction is available and its conditions have to be stated, not
  waved at.** With `g_0 ∈ [δ, 1 − δ]` (Theorem 1's own assumption), a bound sequence eventually
  below `δ` — which `g_bounds="auto"` supplies, since it is `5/(√n·log n) → 0` — and `ĝ`
  consistent **in sup norm**, the clipping event has probability tending to zero, so the
  constrained and unconstrained estimators coincide with probability → 1 and the expansion
  transfers on that event. The sup norm is stronger than the `L₂` conditions Theorem 1 assumes and
  a user-set fixed bound above `ess inf g_0` fails the second condition outright; both are
  restrictions to record, and they are now rows in the assumption matrix rather than a paragraph
  here.
- **The gate is wider than the review stated, and this is the part no document had.** `ĝ` is not
  the only truncated mechanism-side object: equation (10)'s covariate divides by
  `ReducedSet.bounded_gr1(bounds)`, and `g_{r,1}` is a *regression of an arm indicator on `Q̂`*
  whose boundedness away from zero has **no counterpart in Theorem 1's assumption list** — `g_0 > δ`
  does not imply it. `weak-overlap`'s `min gr1` of `0.000` is that bound binding. So the condition
  is "none of the three truncations is active", and the third is an assumption about an estimated
  object rather than about the law.

**What closed it.** Three things had to agree on one sentence: this page, the concordance, and the
estimator's own output. The first two were done in the revision that opened item 25 — the table
above, and the matrix rows behind it — and the third is the **witness**, which landed with
[C1](#the-work-in-five-pieces-and-eighteen-pull-requests). `CorrectionRow.margin` covered the targeted
mechanism at the exit and **nothing on a fit covered the initial mechanism or `g_{r,1}`**, so a user
could not ask a fit which side of the contract it was on without recomputing `clip share` the way
`bench_drtmle.py` does. Now `CorrectionCheck.contract` answers it, `initial_clipped` and
`gr1_margin` are the two columns it needed, `correction_check().summary()` names the label and its
three numbers, and `bench_drtmle.py` reads them off the fit rather than recomputing one of them —
two implementations of "which side of the line is this" being one more than the question can stand.

**And the witness immediately made the condition's second bullet wrong, which is the best thing it
could have done.** The share is not near zero at the exit; it is a sixth to a third on the easiest
process here, for a reason that is about a degenerate covariate rather than about overlap. That is
recorded above and in [the log](drtmle/investigation-log.md#what-c1s-witness-measured-on-its-first-run),
and what it leaves for [C3](#c-the-demonstration) is a *reporting* decision rather than research:
gate 1's clause 0 has to be read as a share per cell, and a cell's coverage number is then partly
evidence about the constrained rendering. Deciding how to report a mixed cell is C3's, before its
dispatch and not after it.

**C3 has taken that decision, before its dispatch**, and it is [the validation plan's fourth
operational rule](drtmle/validation-plan.md#four-rules-that-make-the-gates-operational): the
**pooled** number stays primary and is what gate 1's clauses 5 and 6 read, the **share** is what
clause 0 reads, and coverage within the theorem-side and bound-active draws is reported *beside*
them as description. Neither stratum may be quoted as the theorem-backed estimator's coverage —
the contract label is a post-fit property of the draw, so conditioning on it selects a non-random
subset exactly as excluding invalid fits does, and that is [stop-ship 15](#stop-ship) with a
number attached. What the strata are for is the one question the share cannot answer: whether the
two populations behave differently at all.

**What is *not* claimed.** That the constrained expansion has been derived — it has not, and option
one stays available to anyone who wants the bound-active regime inside the theorem rather than
beside it. And that a bound-active fit is wrong: B1b's measurements say the opposite, which is why
the third option reads *empirically supported and outside the theorem* rather than *unsupported*.

#### C. The demonstration

**Closes item 3, in five pull requests, all landed: [C1](#what-c1-landed),
[C2](#what-c2-landed), C3a, [C3b](#what-c3b-repaired) and [C3c](#what-c3c-measured), the study.
Item 3's gap is there and large, so the demonstration is made and `DRTMLE` does not clear gate 1.
**Item 13 does not close here** — C3c gave it a number, at one reduction learner and as one
aggregate, and [piece E](#e-what-c3c-handed-back) is where it closes either way.** A coverage pilot over the off-diagonal of
the misspecification grid put `TMLE` and `DRTMLE` at 0.958 apiece in one cell and 1.000 in the
other — no gap to close. The diagnosis is understood: a correctly specified *parametric* nuisance
converges at `n^(−1/2)`, so `R₂` is `O(n^(−1))` and the product condition never binds. There was
nothing for the variant to fix.

**That has now happened twice, for a different reason each time, and the pair is the thing to
carry forward.** The Tier-1/Tier-2 construction below exists to answer the paragraph above, and
[C3a's pilot](#what-c3as-pilot-measured-and-why-the-study-is-not-dispatched-behind-it) returned
"no gap" again — this time because the targeting step removes what the design injected, not
because a nuisance converged too fast. Both are the *study* failing to enter the regime rather
than the estimator failing in it. **It did not happen a third time**: C3c entered the regime and
produced the gap, so the tally stops at two — though `g-drift`'s entry column is the same shape one
level down, which [lesson 17](drtmle/investigation-log.md#what-the-sizings-got-wrong) records. A
third recurrence would have been evidence that this regime is hard to
reach on purpose rather than by accident, and [lesson
14](drtmle/investigation-log.md#what-the-sizings-got-wrong) is the distilled form of what went
wrong the second time. `tests/e2e/test_coverage_slow.py`'s `TestDoublyRobustInference`
guards what it can — that the point estimate is still doubly robust, that the interval does not
*cost* coverage, that the standard error matches the spread of the estimates — and says in its own
docstring that it is not a demonstration.

The remainder is `R₂ = ‖ĝ − g₀‖ · ‖Q̄̂ − Q̄₀‖` and a `TMLE` interval needs `√n · R₂ → 0`, so the
regime wanted is one where that product does *not* vanish fast enough while one nuisance is still
consistent. The full specification — the two tiers, the drift-coefficient calculation, the
regime-entry columns, the `P₀D̂` evaluation convention, the reporting schedule, the sizes and
replications, and the frozen decision rules — is in [the validation
plan](drtmle/validation-plan.md#5-the-controlled-study-piece-c), and **what the cells actually are
is [the design note](drtmle/coverage-study.md)**: the committed drift coefficients, the constants
behind them, and the re-timing. What belongs on this page is why
each part is there and what it would take to *not* believe the result:

- **Both off-diagonal cells**, not one. `Q̄` right and `g` wrong, and `g` right and `Q̄` wrong.
  `DRTMLE` should hold nominal in both; `TMLE` should fall short in at least one. One cell is an
  anecdote, because which nuisance is wrong is the whole axis.
- **A prescribed rate, not "a flexible learner in enough dimensions".** That phrase names the
  property wanted and no way to know it was achieved; a Super Learner's realised rate is neither
  identified nor reproducible, so a gap it produces could as easily be finite-sample instability
  as the intended drift. **Tier 1** is a test-only nuisance-injection interface handing the
  estimator `Q̂ = Q̄₀ + n^(−α)·h_Q` while `ĝ → g₁ ≠ g₀`, and the mirror — the only construction in
  which "the intended asymptotic regime was entered" is true by definition, which makes it the
  right place to read item 13's remainder off, and not an applied claim. **Tier 2** is a series,
  spline or histogram regression with a smoothing sequence chosen in advance, and it is the
  demonstration. The Super Learner belongs in the applied stress tests that come after.

  **Both tiers have landed and the split between them is sharper than "two ways of being slow".**
  Tier 1's remainder is a *quadrature* — both nuisances are prescribed functions of `W`, so `R₂` is
  integrated rather than estimated and `n^α R₂ → c` is an identity a test can assert. Tier 2's is
  not: `P₀D̂` at *fitted* reduced regressions needs their values on covariates no fold trained at,
  which is the fold-retained nuisance object §5 asks for and which
  [C2](#what-c2-landed) built as `DRTMLE(evaluation=…)`. So Tier 1's exact remainder is the
  regime-entry column beside Tier 2's rather than a substitute for it, and the two are checked
  against each other where both are computable — two routes to one population integral sharing no
  code.

  Tier 2's own drift coefficient is a **prediction** rather than an identity, which is the honest
  difference: Tier 1 normalises its injected shape so the coefficient comes out at a declared value,
  and here the estimator's bias is what it is. What the study reports is the measured `n^α R₂`
  against the committed one. On one draw at `n = 600` that came out at `0.407` against `0.389` and
  `0.370` against `0.410`, which is §5's *"verify empirically"* rather than an `L₂` rate.
- **A nonzero drift coefficient, chosen analytically and verified.** `α` is a knob to report, not
  a threshold to hit, and the first review's `α < 1/4` is the familiar bar for the
  *both-consistent* product condition — sufficient here rather than necessary, since in an
  off-diagonal cell the misspecified nuisance's error is `O(1)` and `√n·R₂ ≍ n^(1/2−α)`. But
  `α < 1/2` is not sufficient either, and this page previously said it was: **the remainder is an
  inner product, not a norm.** With `Q̂_a − Q_{0,a} = n^(−α)h_a`, the drift is `n^(1/2−α)c_a` with
  `c_a = P_0[(g_{1,a} − g_{0,a})/g_{1,a} · h_a]`, and `c_a` can vanish because `h_a` is orthogonal
  to the misspecification weight even with `‖h_a‖ > 0` — and `c_1 − c_0` can vanish in the ATE
  while both arm coefficients are nonzero. So choose `h_a` so that `|c_a|` and `|c_1 − c_0|` are
  bounded below, commit the calculation with the design, and verify `n^α·R₂ → c` empirically
  rather than inferring the regime from an `L₂` rate.

  **Done, and the construction is worth one line because it makes the proviso structural rather
  than checked.** `h_a` is aligned with the misspecification weight and normalised by the quadrature
  that defines `c_a`, so the coefficients come out *at the design's declared values* — and the arms
  are given **opposite signs**, which makes `c_ATE = c_1 − c_0` a sum of magnitudes and cancellation
  impossible rather than merely unlikely. Committed at `c_ATE = 0.40`, verified against
  `DGP.expectation` — the same Sobol rule the truth is integrated with, so a coefficient and the
  coverage it explains cannot disagree through two quadratures.
- **Coverage against its Monte Carlo standard error**, over replications, with "compatible with
  0.95" defined operationally and a minimum replication count so that a wide interval cannot read
  as success. `CLAUDE.md`'s rule applies with force: never assert coverage on a single fit, and
  size the replication count to the gap being resolved.
- **A size trend** — three sizes if the budget reaches, since two are suggestive and three carry a
  rate. But **coverage need not deteriorate monotonically**: use the slope of the root-`n` bias and
  the remainder as the primary rate evidence and treat a monotone coverage decline as supportive.

**Item 20 was a design input here and B1a is what removes it.** Before the fix, roughly a quarter
of draws report a curve that is not centred — on `nonlinear_dgp`, with good overlap, not in the
weak-overlap cells where this page otherwise leads a reader to expect invalid fits — so the
invalid share would have been ~25% in the cells the demonstration turns on. A coverage number
computed over the surviving three-quarters is conditional on a non-random subset, selected on a
diagnostic correlated with the fit having gone wrong, and reporting it as *the* coverage would be
the same class of error as reporting a per-protocol analysis as intention-to-treat. Of the three
ways out — count an invalid fit as a miss, exclude it and report the exclusion rate beside every
number, or hold the study — **the third was the honest default and is now simply the plan**. The
rule still has to be written down for the residue, before the numbers exist, and it is: the
primary report counts an algorithmically invalid fit as a failure of the procedure, with the
sensitivity to the other two accountings reported beside it.

**Statistical validity and product usefulness are two decisions and the release rule now says
so.** A correctly implemented `DRTMLE` may satisfy its expansion and attain nominal coverage while
`TMLE` under-covers by only 0.03 at reachable sizes. That is a validated method with a modest
finite-sample advantage — two conclusions, not one, and the previous single rule made the 0.05
coverage gap a condition of *correctness*, which it never was.

- **Gate 1, statistical validity**: theorem concordance closed (item 21 included), zero
  state-identity failures, final scores valid, corrected remainder trending to zero with neither
  appendix branch hidden by cancellation, `se` ratio in `[0.90, 1.10]`, coverage compatible with
  0.95 in **both** off-diagonal cells, the conclusion reproducing in the second seed batch, and —
  new with this revision — **the contract frozen and its truncation condition reported per cell**,
  so that a cell whose bound is active is read as [item 25](#the-supported-contract-and-item-25)'s
  third option rather than as evidence about the theorem.
- **Gate 2, practical release value**: a predeclared, practically meaningful improvement over
  `TMLE` — the proposal is a shortfall of at least **0.05** in at least one cell with the Monte
  Carlo interval on the *difference* excluding zero — an acceptable invalid-fit rate, acceptable
  cost, and a benefit that persists in at least one applied stress setting.

The rules may be changed before the final run with a written reason. They may not be changed after
it.

One trap in building it, already met once: `tests/e2e/test_double_robustness.py`'s "correct" cell
is an **oracle** (`OracleOutcomeContinuous`, `OracleTreatment`), which makes the good nuisance
exactly right, `R₂` exactly zero and `TMLE`'s interval already valid. The gap opens only where the
good nuisance is *estimated*.

**"No gap found" remains an honest outcome, and the review sharpens what it obliges.** The
existing rule is to say so in the README rather than keep looking. The addition: in that event
`DRTMLE` does not become a production feature by default — it stays experimental or leaves the
public API until some operating regime is demonstrated. A variant that ships with "we looked and
found no case where this helps" in its own docs is not a neutral state. That is gate 2 failing
while gate 1 passes, and it is a real and reportable outcome rather than a defeat.

**What it costs, re-timed rather than inherited.** This paragraph read *"a `DRTMLE` fit is 43s at
`n = 1,200`, so two cells by two sizes by 250 replicates is ~2,000 fits, ~24 hours serial"* — a
figure measured before piece B1b and before the exit criterion item 7 replaced, which the ordered
list flagged as the thing to redo first. Redone with C1, on a four-core container:

| | measured |
| --- | --- |
| an ordinary `DRTMLE` fit, `nonlinear` at `n = 1,200` | **5.6s**, against the **43s** on record at the same size — a factor of 7.7, which is the *"seventh of the wall clock"* B2b's dispatch predicted |
| the same at `n = 400` | **16.4s** — *slower* at fewer rows, which `test_drtmle_fit.py` already records: noisier nuisances loosen the coupling and lengthen the loop |
| a **Tier-1** pair — `TMLE` and `DRTMLE`, injected nuisances | **1.2s per fit**, at `n = 300` and at `n = 600`/`1,200` alike |
| the Tier-1 pilot, 2 cells × 3 sizes × 50 replicates | 300 draws, 600 fits, under an hour at `jobs=2` |

A Tier-1 fit is cheap because its primary nuisances are *function evaluations* rather than learner
fits; what it pays for is the alternation, which barely depends on `n`. **Tier 2 was expected not to
be**, since its nuisances are fitted and that is what the 43s was measuring — so C2 re-timed before
re-scoping exactly as C1 did rather than inheriting either number, and the answer is that it is
cheap too:

| | measured |
| --- | --- |
| a **Tier-2** `DRTMLE` fit, `q-drift` at `n = 600`, with a 2,000-row companion | **5.4s** |
| the same at `g-drift` | **7.4s** |
| the Tier-2 harness at `--sizes 300 --replicates 2 --evaluation-n 800` | 4 fits in 9s, **1.7s** median |

The additive smoother is cheaper than the boosting library the 43s was measuring, so the frozen
study is affordable at either tier and the workflow's 300-minute cap is generous rather than tight.
What the companion costs is a prediction per fold per nuisance per round and no further learner
fit, and it scales with `--evaluation-n` rather than with `n`.
`.github/workflows/drtmle-coverage.yml` is the dispatch-only workflow, a `matrix:` over the cells;
the nightly tier must not absorb it.

##### What C3c measured

*The study. Two dispatches at 250 replicates, seeds `20250801` then `20250802`, both cells and
`600 / 1,200 / 2,400`, on `main` at `0033c82` with no code change before or between them — 3,000
draws and 6,000 fits. Closes item 3 by **measuring** it, and gives item 13 its first number
without closing it. [The
numbers](drtmle/investigation-log.md#what-the-c3c-dispatch-measured) are in the investigation log
and [the reading](drtmle/coverage-study.md#what-the-study-measured) is in the design note; what
belongs here is what they do to this page.*

**The design worked, and that is the first thing to say because three revisions were spent on it.**
Conditions 1 and 2 pass in all four cell-runs, `n^α R₂(Q̄*)` landing within `0.95x`–`1.01x` of its
committed value at the largest size. Twice before, a coverage study here failed to *enter* its
regime; this one entered it.

**The gap is there and it is large.** `q-drift` at `n = 2,400` reads `TMLE` `0.532` / `0.472`
against `DRTMLE` `0.844` / `0.848`, a paired `+0.312 ± 0.031` and `+0.376 ± 0.033` — six times
gate 2's predeclared `0.05`, with the interval on the difference excluding zero in both batches.
Item 3 has its answer to the question *"is `DRTMLE`'s interval ever better than the one `TMLE`
already reports"*, and it is yes, decisively, in the regime the variant was built for.

**And `DRTMLE` does not attain nominal coverage anywhere in the study.** The best reading in either
cell, at any size, in either batch is `0.880`. Gate 1's clause 6 fails, and so do clauses 3, 4 and
5. **Two measured quantities account for the shortfall, and this revision withdraws the claim that
they are different repairs.** The first is item 13's: `√n R_remaining` is **flat** —
`1.43 / 1.26 / 1.25` in `q-drift` and `4.13 / 4.12 / 4.83` in `g-drift`, reproduced within noise by
the second batch — where Theorem 1 assumes it negligible. The second is the interval's width: the
`se ratio` reads `0.903` in `q-drift`, so the interval is about 10% short of the spread it is
covering. Three revisions of this page attributed the second to `σ²ₙ` "treating the reduced
regressions as known" and split the eleven points about six to one and five to the other. **Both
halves of that are withdrawn** — the formula is Theorem 1's own, the same premise is behind both
numbers, and the split is an artefact of the order the two counterfactuals were applied in;
[the withdrawal](#the-se-shortfall-is-a-symptom-and-what-withdrawing-it-costs) carries the
arithmetic. What stands is that the shortfall is measured, that `g-drift`'s `se ratio` is `1.157` on
the other side of one, and that **the bias is the larger term under either ordering**.

**So item 13 has its first number and stays open on it**, which is the correction this revision
makes to the one before. What the design note called in advance a *finding rather than a fault in
the design* is real and is the result: at these sizes and at `glm` reductions the estimator is
outside the assumptions its own guarantee needs. What does not follow is that the condition is
unmeetable, because the run that tested Theorem 1's conclusion did not establish Theorem 1's own
premise — `reduced regressions consistent` was `unverified` before the dispatch and is `unverified`
after it. [Piece E](#e-what-c3c-handed-back) is that question, with its candidates ranked.

**One number from the study is exploratory and is flagged here because it reads like a conclusion.**
Extrapolating `DRTMLE`'s `√n` bias onto the remainder's level suggests the two meet near
`n ≈ 4,000`–`5,000` at an implied coverage of `0.87`–`0.88`. It rests on a trend the harness itself
reads `unresolved`, and it is **not** a ceiling: the same section fits `R_remaining ~ n^(−0.59)`,
which makes `√n R_rem` itself decline like `n^(−0.09)` rather than sit on a floor. A slow decline
and a plateau are indistinguishable over a fourfold `n` and are not the same claim. Nothing on this
page depends on the number, and no gate is read from it.

**Two things did not go wrong and both are worth recording.** **Zero state-identity failures across
all 6,000 fits** — every invalid fit in the study is a `score` failure, which is the distinction
[B1a](#b1a--the-identity-and-safety-patch) was worded for, doing its work at scale for the first
time. And the study **reproduces**: gate 1's clause 7 passes, with the `se ratio` and the
regime-entry column agreeing to the digit across independent seeds.

**Two labels on the study are narrower than the clause that reads them, and both are in the design
note already rather than new here.** *Batch B is the fully fresh confirmation and batch A is not* —
the two `SeedSequence` streams are prefix-stable, so A shares the **pilot's data seeds** while
drawing its own fold splits, and the pilot is what forced the design change C3b landed. Both
batches agree, so this does not weaken the result; it means the independent confirmation is B, and
A is a rerun of adapted-on draws under a fresh split. *And every cell-run is `BOUND-ACTIVE`*, at
`1.2%`–`8.8%` of draws, where [the harness's own header](../benchmarks/drtmle_coverage.py) says
*"the cells here are designed to be inside it"*. The column did its job and the design did not meet
its intent: a pooled number over mixed draws is the right primary for the estimator as shipped and
is **not** a clean reading of the theorem-backed one. Separating those is [E4](#e-what-c3c-handed-back).

**One reading is narrower than it looks.** `g-drift` has `DRTMLE` resolvably *worse* than `TMLE` at
`n = 600` in both batches. The entry column says why: the correction removes 3–8% of the targeted
remainder there against 70% in `q-drift`, so that cell is one the *plain* estimator's regime was
entered in and the *corrected* estimator's was not. Conditions 1 and 2 are read on `TMLE` by
construction —
[lesson 17](drtmle/investigation-log.md#what-the-sizings-got-wrong) is that distinction as a rule,
and it is [lesson 14](drtmle/investigation-log.md#what-the-sizings-got-wrong)'s shape a third time.

##### What C3b repaired

*The design C3a's pilot found wanting, in both tiers, plus the instrument that says a dispatch may
go. Closes no numbered item, which is why it is its own pull request — [the design
note](drtmle/coverage-study.md#the-repair-and-what-would-say-each-half-of-it-is-wrong) carries the
numbers and what belongs here is what they do to the piece.*

**The repair turned out to be a construction rather than a projection, and that is the finding.**
The pilot proposed projecting out the component the fluctuation absorbs and renormalising, and
warned that a one-dimensional projection removing 95% of a quantity might be treating a symptom.
Writing the algebra out says otherwise: eliminating `ε` through the fluctuation's own score leaves
the estimator's bias as `b_a = P₀[v_a h_a]` against a computable weight, so **`b` is a linear
functional of the injected shape exactly as `c` is** — a coefficient a design can be *built* to hit,
by a 2×2 Gram solve in the span of the two conditions' representers, with the old design as the
one-condition special case. The decomposition was run first as the pilot asked, and it closes as an
identity rather than as an approximate accounting: the fitted `ε` agrees with the population one
within a standard error at every size in both cells.

**Tier 1's root-`n` bias now reads `+1.93 / +2.17 / +3.31`** where the pilot read
`−0.22 / −0.56 / +0.11`, and `R₂(Q̄*)` lands on its declared coefficient at every size. That is
pre-flight condition 1, and it is the sizing paragraph's own arithmetic restored to the column it
was always about rather than discarded.

**Three of the pilot's readings move, and two of them sharpen the failure it found.** The
plug-in-to-targeted ratio was **436** and not twenty — the pilot could bound it, not measure it, and
its measured column was consistent with zero. The design's opposite-arm signs, which make `c_ATE` a
sum of magnitudes and cancellation impossible, **do not survive targeting**: both targeted arm
coefficients came out positive, so `b_ATE` had been a *difference*. And `g-drift`'s corrected
remainder appeared not to rise — re-measured at 12 draws it read `+2.85 / +3.30 / +2.62` against
the pilot's `4.17 → 3.91 → 5.07`.

**That third reading did not survive the dispatch, and this paragraph used to overstate it.** It
said condition 3 *passes in both cells*, which the pre-flight table it summarises never said —
that table reads **`unresolved` everywhere**, on Monte Carlo errors of `±0.42` to `±0.83` at 12
draws. [C3c](#what-c3c-measured) read the same column at 250 draws and got `4.13 / 4.12 / 4.83` in
`g-drift`, back at the pilot's level and not falling. So the 12-draw re-measurement was inside its
own error of the pilot's, the correct 12-draw statement was *not resolvable at this count*, and
condition 3 is the one pre-flight condition the study **failed** rather than passed. `unresolved`
and `passes` are exactly the distinction the verdict table was built to keep, which is what makes
collapsing them here worth correcting rather than deleting.

**The live alternative is closed in the tier and open in one cell, which is not what either
outcome was expected to look like.** The design note refused to talk itself out of the possibility
that *no* injection into a single nuisance produces a first-order shortfall, in which case Tier 1
is a remainder anchor and the repair is a scope correction. That is decided by whether the targeted
weight is degenerate, and it is not: `‖v_a‖ = 0.070` at both arms, measured rather than asserted.
But `g-drift` **cannot** hold the declared drift, and the constraint is *positivity* — it perturbs a
probability rather than a regression with a declared support, the fluctuation absorbs 92–95% there,
and a surviving `0.40` needs `ĝ` to reach `−0.16`. It is declared at `0.10`, the largest value whose
regime-entry column stays put. So **`q-drift` is the cell a gate-2 shortfall is claimed in** and
`g-drift` is where `DRTMLE` is checked to hold nominal under a drift. That is a scope statement on
the face of the design and a property of the estimand's setting, not of the instrument.

**Tier 2's half was a different repair from the one planned, twice over.** It never had Tier 1's
problem at all: its two coefficients agree to five figures, because both its error shapes are
linear in independent standard normals and so have population mean zero — which is what the
fluctuation's step is driven by. That is why Tier 2 produced a gap in the pilot while Tier 1 could
not. What was wrong there is a *reading*: `0.59`–`0.68` was taken as drifting upward and put the
**exponent** in question, and re-measured at the targeted column it is `+0.62 / +0.59 / +0.62` — a
spread of `0.06`, stable, at `1.58×` the prediction. So `β` stays where it is, which is what keeps
the two tiers about one regime.

**Then the obvious repair — shrink the bandwidth until the leading-order formula holds — was run
and falsified.** Scanned over `c_h` of `1.15 / 1.00 / 0.90 / 0.80 / 0.70`, the ratio goes
`1.61 / 1.78 / 1.91 / 2.05 / 2.21`: it **rises** as the bandwidth falls, which is the opposite of
an `h⁴` truncation error and identifies the omitted term as variance-side. Both nuisances are
fitted on the same rows, their errors covary, and that covariance does not shrink with `h`. So no
bandwidth makes the prediction correct, `c_h` stays where it was committed, and what moves is the
**number the pre-flight reads against** — a measured constant beside the analytic prediction, which
is the difference the two tiers already had.

**And the three pre-flight conditions are an instrument now rather than a paragraph** — a verdict
table the harness prints last and the workflow's header says to read first, on the estimator each
condition is about. **Conditions 1 and 2 pass in all four cells**, Tier 1's tightly (`+0.4003`
against a declared `+0.4000` at the largest size). **Condition 3 is `unresolved` everywhere**, and
that distinction is the instrument's own: `P₀D̂` is a quadrature whose error lands directly in each
replicate's remainder and `√n` multiplies it, so at 12 draws every reading is inside its own error
of every other. *Not resolvable at this count* and *failed* are different things, as are *not
measurable* and *failed*; separating them is what C3c's dispatch was for — and
[it separated them](#what-c3c-measured): at 250 draws the errors fall by a factor of four and the
reading is that the quantity is **flat**, in both cells and both seed batches.

##### What C1 landed

The instrument, Tier 1, and the one thing on C's path that is a library change rather than a
benchmark. Four pieces, of which the last was not planned as a finding:

- **the Tier-1 injection**, `benchmarks/drtmle_injection.py`: two scikit-learn-shaped learners in
  the shape `tests/conftest.py`'s oracles already establish, the two off-diagonal cells, the drift
  coefficients committed and verified against `DGP.expectation`, and `exact_remainder`. Three
  decisions in it are load-bearing and each is on the module: the base law is the *easy* one, chosen
  for **overlap** rather than difficulty, because the misspecification is prescribed and the law's
  only job is to put the cells inside the contract; the outcome scaler is **declared** through
  `q_bounds=` rather than recovered, since `OracleOutcomeContinuous`'s recovery carries an
  `O(n^(−1/2))` error and that is *the same order as the injected drift*; and `h_a` is aligned with
  the misspecification weight with the arms given opposite signs, so no coefficient can cancel;
- **the harness**, `benchmarks/drtmle_coverage.py`: both estimators on the same draw at the same
  injected nuisances, so the shortfall is **paired** and gate 2's `0.05` is resolvable at 250
  replicates rather than at an order of magnitude more. `EstimandSummary` is reused for every
  standard column, and `CoverageStudy` deliberately is not — it swallows exceptions, keeps no
  per-replicate row, cannot pair two estimators on a draw and carries no per-fit diagnostic;
- **item 25's witness**, `CorrectionCheck.contract` with `initial_clipped` and `gr1_margin` beside
  the `margin` B1b left, on the face of every fit and in `to_frame()`. `bench_drtmle.py` now reads
  its `clip share` off it rather than recomputing — two implementations of "which side of the
  contract" being one more than the question can stand — and the column came out numerically
  unchanged, which was checked before the duplicate went;
- **and the finding the witness produced on its first run.** A sixth to a third of well-overlapped
  draws exit **bound-active**, with the initial mechanism never clipping, because equation (9)'s
  covariate `Q_r/g*` vanishes where the outcome regression is right — so its score's root is an
  `epsilon` of 24 on the logit scale and drives rows to both bounds. That is item 4 with the
  nuisances swapped, it makes [item 25](#the-supported-contract-and-item-25)'s second bullet
  overstated, and it leaves C3 a reporting decision. [The
  measurement](drtmle/investigation-log.md#what-c1s-witness-measured-on-its-first-run).

What it did **not** land, by name: `R_remaining` and the two appendix-B branches, which need
`P₀D̂` at the *fitted* reductions and so C2's retained per-fold nuisances; and any coverage verdict,
since `--tier 2` was refused rather than approximated and Tier 1 is not the demonstration. Both of
the first two landed with [C2](#what-c2-landed); the third is still true of C1's numbers and of
C2's.

##### What C3a's pilot measured, and why the study is not dispatched behind it

*The freeze landed and the pilot ran — 600 fits, both tiers, both cells, 50 replicates at three
sizes. It did what a pilot is for and the answer was not a constant to re-tune.* [The
numbers](drtmle/coverage-study.md#what-the-pilot-measured) are in the design note; what belongs
here is what they do to the piece.

**Tier 1 cannot produce the gap it was built to produce, and that is structural.** Its injection is
exactly what it committed to — `n^α R₂` reads `+0.4000` at every size — and the plain interval does
not under-cover anywhere: `TMLE` covers at `0.90` to `1.00`, over-covering at `se` ratios up to
`1.52`, against a design that predicted `0.87 / 0.86 / 0.81`. The cause is a **reading** and not an
arithmetic slip. `exact_remainder` integrates the *plug-in* remainder `R₂(Q̂)`, which its docstring
says; a fit's bias is `ψ̂ − ψ₀ = (Pₙ − P₀)D* + R₂(Q̄*)`, the same expression at the **targeted**
regression. `benchmarks/drtmle_tier1_bias.py` evaluates both on the same rows of the same fits and
the mean bias tracks `R₂(Q̄*)` — `−0.004`, `+0.011`, `−0.002` — while `R₂(Q̂)` sits at `+0.081`,
`+0.068`, `+0.057`. A factor of about twenty. So Tier 1 injects its drift into `Q̂`, where the
fluctuation's own free parameter absorbs it, and no choice of `c` changes that.

**Tier 2 has a gap in one cell of two, under a regime it did not commit to.** At `n = 2,400`
`q-drift` reads `TMLE` `0.540` against `DRTMLE` `0.760`, a paired `+0.220 ± 0.072` — the shape the
variant exists for. `g-drift` reads `0.700` against `0.740`, and at `n = 600` `DRTMLE` is *worse* by
`−0.120 ± 0.055`. Its realised `n^α R₂` is `0.59`–`0.68` against the committed `0.389`/`0.410` and
drifting upward. Read against the rules this pilot **fails gate 1** at clauses 4, 5 and 6.

**No defect surfaced, and that is worth stating plainly.** `identity` is `0` across all 600 fits, so
clause 2 holds and [B1a](#b1a--the-identity-and-safety-patch)'s distinction is doing its job the
first time anything read it. Nothing here is evidence against the estimator; what it is evidence
about is the **instrument**. That is why the 250-replicate dispatch is not queued behind it: §5's
rules may be changed before the final run and not after it, so a study run now would be a study of a
design whose premise its own pilot contradicts — and this is the one run on this page whose cost
makes redoing it a decision rather than an errand.

**The decision is to repair the design**, rather than to run it as it stands or to report "no gap"
a second time, and C3 splits into three for the reason B2 split into two: the repair precedes the
dispatch and closes nothing on its own. [The repair and what would say each half of it is
wrong](drtmle/coverage-study.md#the-repair-and-what-would-say-each-half-of-it-is-wrong) is the
written reason §5 requires for moving a rule before the final run, and it is two halves because the
tiers failed differently:

- **Tier 1's injection is absorbed by the targeting step.** The score equation constrains a
  `g₀/ĝ`-weighted offset of `Q̄* − Q̄₀`, the estimator's bias is the *unweighted* one, and the design
  chose `h_a` to make the **plug-in** remainder large — which is no condition on the targeted one.
  The repair is one further linear condition on a function already chosen by quadrature. **What
  argues against it**: a single `ε` per arm removes a one-dimensional component and the measurement
  says essentially all of `R₂(Q̂)` went, which is more than a projection explains — so the first
  thing to run is a decomposition of the existing injection and not a new one. And the live
  possibility that an off-diagonal cell *cannot* produce a first-order shortfall, because a `TMLE`
  with one consistent nuisance is consistent and that is double robustness working: in that case
  Tier 1 is a remainder anchor, was never a demonstration, and the repair is a scope correction.
- **Tier 2 enters a regime, but not its own.** It does produce the gap — `+0.220 ± 0.072` in
  `q-drift` — while its realised `n^α R₂` drifts upward across sizes against a committed constant.
  **What argues against a bandwidth fix**: `g-drift`'s *corrected* remainder rises, which is item
  13's condition failing rather than this design's, and re-tuning a smoother's bias does not
  obviously repair it.

**Three pre-flight conditions**, none of which is a coverage study and all of which are minutes:
`R₂(Q̄*)` at the declared `n^(−α)c`; the realised `n^α R₂` stable across sizes; `√n R_rem` falling
in both cells. §5 now requires the first [by
name](drtmle/validation-plan.md#verifying-the-regime-was-entered), because verifying only the
plug-in coefficient establishes that the injection is what it says and not that the regime was
entered.

**This is the second time a coverage study here has returned "no gap", for a different reason each
time** — the first was a parametric nuisance converging at `n^(−1/2)` so the product condition never
bound, and this one is the targeting step removing what was injected. Both are the study failing to
*enter* the regime rather than the estimator failing in it. A third would be evidence that the
regime is hard to reach on purpose, and [lesson
14](drtmle/investigation-log.md#what-the-sizings-got-wrong) is the distilled form.

##### What C2 landed

Item 13's instrument, Tier 2, and the columns gate 1's clause 4 reads. **It does not close item
13**, and the distinction is the one A1b's stability half is held to: what landed is the thing that
makes a rate measurable, and a rate is three sizes at 250 replicates, which is C3's dispatch.

- **the evaluation companion**, `DRTMLE(evaluation=…)`, and it is a **library** keyword where §5
  asked for a benchmark-only object. That departure is the piece's one real design decision and it
  is recorded rather than slipped in: retaining the models and *replaying* the alternation outside
  the library is a second implementation of `solve_with_reduction`'s state map, and that map is the
  hard part — the outcome solve applies its tilt once per Newton step and shrinks after each,
  `solve_bounded_mechanism` clips, and the reductions are refit every round, so `Q̄*` is **not**
  `expit(logit Q̄̂ + ε·H)`. A bug in a replay is indistinguishable from a real remainder. So the
  evaluation rows travel **through the same solvers**, which is `Fluctuation.carried` generalised in
  one way: a carried item supplies its own clever covariate, because the evaluation rows' is not the
  fitting rows'. A third diagnostic keyword beside `update_order=` and `reduced_crossfit=`, and for
  the same kind of reason;
- **and it is anchored by an identity rather than by an argument.** Handed the fitting frame back
  as its own companion, fold `k`'s slab must reproduce the production array at the rows fold `k`
  holds out — every initial array, all three reductions, and the *targeted* `Q̄*` and `g*`. That
  fails if a slab is read one fold out, if a round's tilt is dropped, if the companion travels along
  the production covariate, or if a refit is taken at a stale design; two of those were mutated and
  watched to fail. The companion is otherwise **inert**: a Tier-1 study run on the pre-C2 code and
  on this one agreed on 48 rows across 12 fields with zero mismatches;
- **Tier 2**, `benchmarks/drtmle_tier2.py`: both nuisances fitted, the good one an oversmoothed
  **additive kernel** at a committed `h_n = 1.15·n^(−0.125)`, with the drift coefficients computed
  by quadrature at `c_ATE = 0.389` and `0.410` — Tier 1's `0.40`, so the two tiers share the rate of
  the *remainder*, which is what they have to share to be about one regime. **Not** the regressogram
  §5 names, and that is a finding: a regressogram's bias oscillates within a bin, so its `L₂` norm
  is `O(B⁻¹)` while its inner product with a smooth weight is `O(B⁻²)` — and the remainder is an
  inner product, so matching a declared rate with one needs a bin count at which the fit is
  variance-dominated and its remainder is noise. §5's trap, through a second door;
- **the remainder columns**, `benchmarks/drtmle_remainder.py`: `R_remaining` and `R₂` at the fitted
  nuisances, both **exact** given the companion, and the two appendix branches, which are **not** —
  each is a binned quadrature reported at two bin counts with their difference beside it, and a
  branch smaller than that error is reported as unresolved rather than as a number. The `M` terms
  are refused by name, because `(P_n − P_0)` of a fold-conditional function has no single-sample
  rendering under the fold-weighted convention `P₀D̂` is taken at;
- **and the re-timing the page asks for before C is re-scoped.** A Tier-2 `DRTMLE` fit with a
  2,000-row companion is **5.4s** to **7.4s** at `n = 600`, against the **43s** this page costed C
  from and beside C1's **5.6s** for an ordinary fit. Tier 2 was expected to be the expensive tier
  and is not: the smoother is cheaper than the boosting library the 43s was measuring.

What it does **not** land, by name: any coverage verdict and any rate. A single draw's
`R_remaining` is dominated by the evaluation draw's own quadrature error — `sd(D)/√m`, which lands
directly in it — so the number that means something is the replicate mean with its Monte Carlo
error, which needs the dispatch.

#### E. What C3c handed back

**Closes item 13 either way. It was seven pull requests in a line — E0 to E5 — and after E2 ran it
is four that have landed and [three work packages](#the-reboot-and-the-three-work-packages) with a
branch in the middle**, which is a narrowing rather than a rescoping: the same question, with the
route to it made conditional on the one experiment that can answer it.
[C3c](#what-c3c-measured)
measured `√n R_remaining` and found it flat. It did not find out *why*, and the difference is the
whole of this piece: a run whose reduced regressions were fitted by `glm`, whose consistency the
concordance marks `unverified`, tested Theorem 1's conclusion without establishing Theorem 1's
premise. **The honest reading of that study is a falsification of the configuration, not of the
estimator**, and the work below is what tells the two apart.

**This piece is diagnosis before dispatch, and that ordering is the lesson C3a and C3b already
paid for twice.** The tempting next action is the same study at a larger `n` with the same
reductions and the same 2,000-row companion. It is the one action the measurements jointly rule
out: it cannot distinguish any of the three candidates below, and at `n = 4,800` it is
[several times C3c's](#what-c3c-measured) 77–112 minutes a cell.

##### The three candidates, ranked by what the study measured

**They are not interchangeable and the previous revision of this page listed them as though they
were.** Ranked by the evidence actually in the tables:

1. **The reduced regressions are inconsistent at `glm`.** *The strongest, and the cheapest to
   test.* It is the only candidate with a **named unverified premise** — the concordance's
   `reduced regressions consistent` row reads *estimated, unmeasured rates* — and the only one with
   a **matching signature**: what survives in `g-drift` is the outcome-side reduction (`R_Q +0.098`
   against `R_g −0.032`) in the cell whose outcome regression is the deliberately misspecified one,
   which is exactly where `Q_r = E[Y − Q̂ | A = a, ĝ]` is furthest from zero and least likely to be
   linear on a link scale. **E2 was to decide it and did not**: three of its four cells are
   `unresolved` on gate B and the fourth reads `moved`, so the candidate is
   [alive and unestablished](#what-e2-measured-and-why-it-did-not-branch) rather than settled. What
   the run did establish is a **repair with a name** — a rung chosen against a measured ranking
   rather than assumed — and that repair is [E2R](#the-reboot-and-the-three-work-packages).
2. **The construction or the solver adds first-order variation.** `‖Δ_k‖ = o_p(1)` is `unverified`
   and what the A1b dispatch measured is its *consequence* on `ψ` rather than `‖Δ_k‖`; and 99 of
   3,000 fits exit invalid, ten of twelve cell-runs over the 2% bar. Both are effects on the
   interval that are **not** the remainder. **This candidate used to carry a third piece of
   evidence and no longer does**: the `se ratio` of `0.903` was read as a variance estimator that
   treats the reductions as known, and it is not — it is the same unestablished premise as
   candidate 1, seen through the second moment instead of through the mean
   ([the withdrawal](#the-se-shortfall-is-a-symptom-and-what-withdrawing-it-costs)). So this
   candidate is **weaker than the previous revision had it**, and it is now the branch **E3** takes
   only if E2R comes back `equivalent`, or if a repaired remainder leaves the calibration still
   failing.
3. **`n` is simply too small.** The residual candidate, the only one no repair reaches, and the
   only one whose test is the expensive dispatch. It is listed last because it is not *testable*
   until the first two are excluded — a rate read through an inconsistent reduction and a short
   interval is a rate for neither.

**A fourth was proposed and the artefacts contradict it**, and it is recorded here because it is
the one an experienced reader reaches for on seeing `g_{r,2}/g_{r,1}` in a covariate: *a small
`g_{r,1}`, or an active truncation, is what makes `q-drift` fail*. Every C3c fit carries
`gr1_margin` and `initial_clipped` — [item 25's witness](#the-supported-contract-and-item-25),
which is why C1 is its own pull request — and the columns say no. `min gr1 margin` is `0.117` to
`0.343` across all twelve cell-runs and the initial mechanism's `worst clip share` is `0.0000` in
eleven of them; the critique's own read of the manifested rows puts `q-drift` at `n = 2,400` at a
minimum margin of `0.274` with zero clipping activations in 500 fits, and the bound-active exit at
`7/500`, whose `se ratio` is `0.904` against the pooled `0.908`. **The denominator is not where this
failure is, and no further cycle goes on tuning that bound for it.** What that does *not* license is
dropping the columns: they are what makes the sentence checkable, and item 25's scope decision reads
the same witness.

##### The four that landed

**E0 is not a piece**, under the rule the contract row is held to: it is documentation, it closes
no research, and it is separate so that a correction to the record cannot arrive entangled with an
estimator change.

| | scope | what it closes | must not include |
| --- | --- | --- | --- |
| **E0** — *landed* | the record: item 13 reopened, the ceiling demoted to exploratory, the two mechanisms separated, C3b's condition-3 sentence corrected, batch A relabelled, and the four stale status-prose sites. Plus [the study manifest](drtmle/study-manifest.md) | nothing — it is the page saying what the study showed | any estimator or benchmark change |
| **E1** — *landed*, in part | the remainder instrument: `P₀D̂` by deterministic integration — the curve is affine in `Y` and binary in `A`, so two of its three coordinates close in closed form — a nested-grid ladder read off one fit per draw, and the full score row per fit serialised beside the replicates. [What it landed, and what E1b withdrew](#what-e1-landed-and-what-e1b-withdrew) | the *mechanism* half of the quadrature question. **Its two quantitative readings are withdrawn** — the ladder was read as a bound and the variance ratio as a share, and neither is what those statistics are | learner selection; any coverage claim |
| **E1b** — *landed* | the precision half, measured: an **independent scramble per replicate**, which makes the grid's error mean-zero rather than a fixed bias, so it can be estimated *conditionally on each fit* from replication rather than inferred from a successive difference. Every share prints with an interval, and the numbers come from a dispatched run with retained artefacts. [What it measures](#what-e1b-measures) | how much of C3c's across-draw spread was the evaluation rule — the number E5 sizes a replicate count against | learner selection; any coverage claim; any rate |
| **E2** — *landed; ran; did not branch* | **the reference reduction, and it is the branch point.** Build `Q_r`, `g_{r,1}`, `g_{r,2}` at their population limits — a *univariate* projection each, on the companion, injected through `ReductionSpec.refit` at the current targeted pair every round — and refit both cells with them. **A reference is a numerical estimate and not an oracle, so its own fidelity is gated first**: an exact-law control, a validation loss on an independent finer companion, and a randomisation budget — none of them a refinement difference. Then, against a **precommitted** equivalence margin: if the reference does not move `√n R_remaining`, candidate 1 is dead and the learner road is shut, and the diagnosis moves to E3. **`unresolved` is a third verdict and is not a weak "dead"**, and it is the verdict three cells of four returned — on **gate B**, each on a coarser rung beating the shipped one and each on a different reduced regression. [What it measured](#what-e2-measured-and-why-it-did-not-branch) | it was to close whether item 13's failure is a *learner* failure at all, and it **did not**: the branch is unmade and the repair is [a rung selected rather than shipped](#the-reboot-and-the-three-work-packages) | a learner comparison. That is E2b's and only if this fires. Any margin chosen after a number exists. Calling the reference an oracle |

##### The reboot, and the three work packages

**What the reboot changes is not the destination but the *conditionality*.** The old list ran
E2 → E2b → E3 → E4 → E5 as a sequence in which E3 was going to happen whatever E2 said, because the
`se` ratio was believed to be a second mechanism with a repair of its own. It is not
([the withdrawal](#the-se-shortfall-is-a-symptom-and-what-withdrawing-it-costs)), so the work is one
diagnostic, **one branch**, and one confirmatory study — and the branch decides whether E3 happens at
all.

| | what it is | fires when | closes |
| --- | --- | --- | --- |
| **WP1 — E2R** | the repaired reference experiment, and the *only* action that separates reduction error from estimator construction | now. It is the piece's next dispatch | whether item 13's failure is a learner failure at all — the question E2 was built for and did not answer |
| **WP2 — E2b *or* E3** | a feasible-learner comparison, **or** the narrow construction/stability diagnosis. **Not both, and which one is E2R's output** | on E2R's verdict, per [the branch table](#the-branch-e2r-decides) | the reduction learner, or item 15's stability half and item 22's numerical half |
| **WP3 — E4 + E5** | the frozen confirmatory study: the preregistered two-design split, then the fresh two-batch dispatch | only after WP2 resolves. **A coverage number taken before it is not the release number** | **item 13**, and item 3's second half |

**The old letters are kept rather than renumbered**, because
[`docs/drtmle/study-manifest.md`](drtmle/study-manifest.md) cites E3 by name for the 99 replayed
seeds and [the coverage study](drtmle/coverage-study.md) cites E4 and E5 for the design split and
the rate. What moved is when each fires, and E2R is the one genuinely new row.

**WP1 — E2R, one conclusive reference experiment.** The same design as E2, repaired at eight points
and dispatched once — five against E2's own failure, and three more against [a review of the
instrument](drtmle/validation-plan.md#the-decision-protocol-frozen-before-the-dispatch) that found
the first five did not make it confirmatory. **PR #74 is demoted to a pilot** and its seed streams are spent:
the decision run takes fresh ones.

**The instrument for all five has now landed and the run has not happened**, which is this page's
own C1/C3a/B2a pattern rather than a deferral: *the repair precedes the dispatch and closes nothing
on its own*, and a workflow is dispatchable only from `main` — which is the whole of what E2 was
waiting on when the previous revision reported a `404`. [What landed and what a sandbox pilot
moved](#what-e2rs-instrument-landed) is the record; the five clauses below are what it was built
against, and two of them came out of the pilot differently from how they were written.

1. **The rung is selected rather than assumed, and per `(cell, size)` at least.** E2 shipped one
   `spline(16)` for three regressions in four cells; what its intervals reject is *one rung across
   the two cells*, and the per-regression reading is [a point-estimate reading of an interval
   gate](#what-e2-measured-and-why-it-did-not-branch). Selecting independently for `qr`, `gr1` and
   `gr2` is the right *procedure* — it subsumes the case where they agree — but the pilot's winners
   are **not** hard-coded, because a rung chosen on the run that certifies it is a rung that
   certified itself.
2. **Fit, select, audit and evaluate are four blocks, and E2 has three.** Reference `4,096`,
   scoring `8,192`, evaluation `2 × 2,048`, on disjoint scramble streams — which was enough because
   E2 never *selected*: it shipped a rung and used gate B to check it. Turn that ranking into a
   selection and the block that chose the rung becomes the block that certifies it. **The audit
   block must not have chosen what it certifies**, so the scoring block splits.
3. **Two composite gates beside the three component ones.** Gate B scores `qr`, `gr1` and `gr2`
   apart, which is right and stays — it is what detects an individually bad reduction and prevents
   two errors cancelling. But none of the three is what the fit reads: `reduced_correction_parts`
   builds
   `H₂ = g_{r,2}/g_{r,1}` and `H₃ = q_r/g`, per arm, **at the bounded denominators**. So the audit
   gets those two as losses in their own right, computed at the same `g_bounds` the fit used — at
   any other bound it is a loss on a different object — with the denominators' margins and
   truncation rates recorded beside them. *Componentwise risks are theorem-relevant and incomplete,
   not wrong.*
4. **The negative control must be detectably inferior on the metric it is meant to test**, `gr2` at
   `g-drift` `2,400` included, where E2's `bins(8)` read `-1.031e-04 [-4.35e-04, +1.12e-04]` and the
   gate had no teeth. A gate that cannot reject the deliberately coarse arm cannot certify anything,
   and the composite gates get the same clause.
5. **More reference points is a lever and not only a finer rung.** Gate B ranks candidates at a
   *fixed* block size, so "`spline(8)` beats `spline(16)`" says the extra knots cost more variance
   than they buy in bias **at 4,096 points** — the two repairs are one bias–variance trade seen from
   two sides, and only the block size moves the reference towards its limit rather than towards
   internal consistency. It is affordable: the pilot priced a doubling at `25.5s` a fit against
   `13.7s`. **The falsifier is stated in advance**: if doubling the block moves the gate-B ordering
   towards the finer rungs, the pilot's rung was a statement about the block and not about the
   function.

6. **Selecting and deciding are two cohorts of draws and two commands, separated by a commit.**
   A review of the instrument's own pull request found that clause 2 repaired the wrong level:
   four disjoint scramble streams split the *integration* noise, and both risk tables were still
   functions of the same fitted samples, fold splits and draw-specific nuisance states. The draw
   is the independent unit — `draw_risks` averages within one and every interval resamples them —
   so a rung selected across a set of draws and certified on the same set is a data-dependent
   selection assessed on the sample that made it. `--phase select` ranks on the **selection**
   cohort and writes a manifest; the manifest is committed; `--phase decide` fits the **decision**
   cohort at a mapping it can read and cannot choose, and refuses one whose recorded draws are its
   own. The disjointness is checked on the **data seed**, which is the sample, since C3c already
   met the case where prefix-stable streams shared data seeds under different splits.
7. **The fidelity clause passes on a non-inferiority bound and never on failure to show
   superiority.** The same review found that gate B failed only on an interval lying wholly below
   zero, so an imprecise comparison certified the reference by default — and this page's own
   record calls two rungs "genuinely indistinguishable" on exactly that reading. An interval
   containing zero establishes neither equality nor adequate approximation. The clause is now a
   *simultaneous* one-sided lower bound over every competing rung and all five metrics, against
   `δ_metric`: on the two composites, the excess risk that could move `√n R_remaining` by
   `FIDELITY_FRACTION = 1/3` of the margin — gate C's own share, bounding the reference's
   **smoothing** error where gate C bounds its quadrature error — and on the three componentwise
   metrics, `COMPONENT_FRACTION = 0.05` of the negative control's measured distance, since those
   have no tight transfer to the column. The older clauses stay and stay gating.
8. **A missing reading cannot reach a verdict.** `selection_rows` raises where a regression has no
   reading, `Payload.references` raises where a rung was not selected, the manifest is validated
   before anything is fitted, and a cell resting on fewer than `COMPLETENESS_FRACTION = 0.9` of
   its declared draws is `unresolved` and marks the **run** invalid. An invalid run exits
   non-zero. The fallback these replaced was defended as *visible* — E2's shipped rung, named in
   the record — and it was visible in the record and not in the verdict.

*The rule the comparison is judged by does not move.* `EQUIVALENCE_FRACTION`, `BUDGET_FRACTION` and
`PRIMARY_ESTIMAND` are what they were before E2 ran. Every item above is a change to the
**reference** or an **addition to the gates**, which [§8](drtmle/validation-plan.md#8-the-reference-comparison-piece-e2)
permits before a dispatch with a written reason and this list is the reason; and each makes a verdict
*harder* to reach, which is the only direction a gate may be added in after numbers exist. Items 6
to 8 are held to that too, and the one place it bites is clause 7: a non-inferiority margin would
have made the pilot's failing `qr` clause pass, so the superiority clause it repairs was **kept
beside it** rather than replaced. A cell failing only on superiority with every bound clear is a
reportable finding, not an occasion to relax anything.

**Bound: one instrument repair and one decision run.** If the repaired reference still cannot
establish equivalence, E2R reports `unresolved` and **stops being cited as motivation for a
production change**. That is a real outcome and the page will say so. The review's repair does not
buy a second dispatch: it is the same one decision run, on a design that can carry it.

##### The branch E2R decides

| E2R comes back | WP2 is | why |
| --- | --- | --- |
| gates pass, remainder **materially improves** | **E2b** — compare feasible reduction learners, then freeze the winner | the reference is the existence proof; the open question becomes attainability |
| gates pass, remainder **does not improve** | **E3** — the narrow pooled-versus-nested construction and stability test | candidate 1 is dead and the learner road shuts, which is E2's `equivalent` arm reached at last |
| gates pass, reference improves it, **no feasible learner approaches it at `n = 2,400`** | **stop.** Record a negative learnability result | "an oracle would fix it" is not a shippable estimator, and saying so is a result — the same standard `bench_tmle.py`'s no-Rust conclusion is held to |
| gates still **fail** | **stop.** More coverage runs cannot answer this | a comparison at a reference another resolution beats answers for the wrong reference, at any replicate count |

##### What E2R's instrument landed

**The five clauses are built and the decision run is owed.** What is new is a *selected* reference
and a gate with more teeth; what is unchanged is every constant the comparison is judged by —
`EQUIVALENCE_FRACTION`, `BUDGET_FRACTION` and `PRIMARY_ESTIMAND` are what they were before E2 ran,
and [§8's rule](drtmle/validation-plan.md#the-reference-rule-frozen-before-the-dispatch) was not
touched. What *is* new in the plan is a **selection rule**, frozen in the same place and in the
commit before the dispatch, because a rung that is measured rather than shipped needs one.

| clause | what landed |
| --- | --- |
| the rung is **selected**, per `(cell, size, regression)` | `select_rung` — the **coarsest admissible** rung, admissible meaning *no other rung's paired interval lies wholly below zero on any of its metrics*, which is gate B's own clause read on the block that chose. The pilot's winners are inputs to nothing: `reference=` now takes one object per regression, and the mapping the fit is handed comes out of the run's own pass one |
| four blocks, so the block that **certifies** is not the block that **chose** | `Layout` carries `reference`, `selection`, `audit` and `evaluation` on four disjoint scramble streams, all fresh — PR #74's are spent. The run is two passes with a barrier: the control arm ranks, then the reference arm is fitted at what the ranking selected and audited on rows the selection never saw |
| **five** gate-B metrics, not three | `METRICS`: `qr`, `gr1`, `gr2`, and `h3 = q_r/g` and `h2 = g_{r,2}/g_{r,1}` at the divisors the fit uses, at its own `g_bounds`, each divisor's margin and truncation share recorded beside it. A composite is the same held-out risk under the weight `w/d²` — the cross term still vanishes, because both divisors are functions of the conditioning index — so the ranking property carries over intact, and what it *cannot* see is the divisor's own error, which is what the componentwise `gr1` metric is for |
| the control is **detectably** inferior | `bins(2)`, with `bins(4)` and `bins(8)` reported. Both the coarsening and its size came out of a measurement rather than a preference — see below |
| the block size is a declared **lever**, and it has been **pulled** | `--reference-points` at `8,192` rather than E2's `4,096`, because the falsifier — *if doubling it moves the ranking towards the finer rungs, the winner was a statement about the block* — was run and **did not fire**: the selection is rung for rung identical at both sizes. What the doubling did do is resolve the audit, and it is nearly free at four blocks |

**Three sandbox pilots sized the instrument, and between them they moved four of the choices
above.** All tier 2 in `g-drift` at `n = 2,400` — the cell whose gate E2 failed twice — at the
dispatch's own block sizes. **No comparison column from any of them is read**, for the reason E2's
own pilot's were not: a difference under a sizing run is what the rule refuses, whatever it says.

- **The first, at six draws: `bins(4)` was not coarse enough, and only one metric says so.** It is rejected on `qr`, `gr1`,
  `gr2` and `h2` — including the `gr2` case E2 failed on, `+2.076e-04 [+8.70e-05, +3.67e-04]`
  against E2's `-1.031e-04 [-4.35e-04, +1.12e-04]`, so the named repair took — and **not** on `h3`,
  `+7.190e-05 [-1.16e-04, +2.21e-04]`. `h3` divides by a bound-active targeted mechanism, whose
  margin the same run reads at `-0.0135`, so its weight carries the heavy tail `1/g*²` gives it.
  Hence `bins(2)`, and hence the two coarser arms staying as *reported* rows: a clause on the
  coarsest control is a necessary condition on the instrument, and the rung-against-rung intervals
  beside it — differences of `1e-06` resolved at six draws — are what say how finely the audit can
  discriminate.
- **The ranking is taken at a *targeted* state, and the initial pair was the first choice.** The
  initial pair is the one state both arms and every rung share, which is why it was written first.
  The pilot ranked `h3` one way there and the audit's own point estimates the other way at the exit
  state — and `h3`'s divisor is a mechanism whose bound-activity is *made* by targeting, so a
  ranking taken before the alternation ranks at a divisor no fit uses. `RecordingDRTMLE` records
  the **control arm's** exit state, which is candidate-free — the `glm` reduction is the arm under
  comparison, not a rung in the ladder — and targeted. The residue is stated rather than hidden:
  the reference arm's own exit state is not the control arm's, and no selection can be made at the
  state it will be certified at without certifying itself.
- **And the selection rule was a minimax on point estimates and is now admissibility on
  intervals**, which is the same pilot's third finding and the one worth carrying past this run. The
  first rule minimised each rung's worst *relative excess* of the mean risks; on `qr` that bought a
  `0.002` relative loss on the componentwise metric — a `2e-06` difference, **resolved** at six
  draws with its interval clear of zero — for a `0.01` apparent gain on `h3`, whose intervals there
  straddle zero by an order of magnitude. It selected `spline(32)`, and gate B rejected it on
  precisely the difference the rule had discounted. **A selection judged on point estimates cannot
  be certified by a clause read on intervals**, which is
  [lesson 21](drtmle/investigation-log.md#what-the-sizings-got-wrong)'s third part. The rule now
  selects the coarsest rung that no other rung *significantly* beats, so the audit asks whether
  admissibility **replicates** out of sample — a far more useful question than whether a
  point-estimate winner survives an interval.
- **Then twelve draws under the repaired rule, which is the confirmation and was not a clean pass.**
  The teeth clause holds on **all five** metrics — `bins(2)` rejected at `+4.159e-04`, `+6.045e-04`,
  `+3.067e-03`, `+1.702e-03` and `+4.226e-01`, every interval clear of zero, so the clause E2 could
  not satisfy in this cell is satisfiable. The selection finds an admissible rung for each
  regression, `beaten on` `0` throughout. And it **did not replicate on one metric**: the audit read
  `spline(8)` beating the selected `spline(16)` on the componentwise `qr` by `-7.615e-07`, resolved
  because the pairing is tight. **The clause was not relaxed for it** — a tolerance introduced after
  a failure is the one direction a gate may not move — and an `8e-07` difference on a risk of order
  `1e-03` implicates the ladder's *resolution*, which is what the next bullet is.
- **So the block-size lever was pulled rather than held in reserve, and its falsifier did not
  fire.** The same twelve draws at `--reference-points 8192` select **rung for rung the same**
  reference, so the winners are statements about the function and not about the block — and the
  audit then passes on all five metrics, the `qr` difference reading `-3.020e-07 [-7.71e-07,
  +1.46e-07]`, two rungs genuinely indistinguishable rather than one beating the other. It also
  takes `h2`'s divisor back inside its bounds, margin `+0.1636` against `-1.1693`, so that
  composite's weight stops carrying a two-order tail. `8,192` is therefore the dispatch's default,
  at `52.2s` a fit against `45.8s` — nearly free at four blocks, because a fit's cost is now the
  companion's predictions rather than the reference's own least squares.
- **Gate C passes by a factor of sixty** — `0.0069` against `0.5167` — and the four blocks cost
  `57,344` companion rows against E2's `32,768`. The workflow's timeout moves with that and nothing
  else does.

**What none of this does is decide anything.** No verdict here is E2R's: these are one cell at six
and twelve draws, read to size an instrument, and **each pilot's comparison columns exist and are
deliberately unquoted** — a difference read under a sizing run is what §8 refuses, the reason for
every choice above is gate and cost arithmetic, and the dispatch takes fresh seed streams over both
cells and both sizes. **And the bound still holds** — one repair and one decision
run — so an `unresolved` E2R ends the reduction road as evidence rather than earning a third
dispatch.

> **One phrase above has since been withdrawn as a certification, and it is worth reading where it
> was written.** "Two rungs genuinely indistinguishable rather than one beating the other" is a
> fair description of `-3.020e-07 [-7.71e-07, +1.46e-07]` and it is **not** a statement that the
> selected rung is adequate: an interval containing zero says the run cannot resolve the two, which
> is a fact about the run. A review of this instrument found that gate B was built to read it as
> adequacy — it failed only on an interval wholly below zero — so an imprecise comparison certified
> the reference by default. The clause is now a non-inferiority bound against a declared margin
> ([§8](drtmle/validation-plan.md#2-the-fidelity-clause-is-non-inferiority-against-a-margin-declared-in-advance)),
> and this pilot's `qr` reading would now have to *clear* `δ_metric` rather than merely straddle
> zero. The block-size conclusion above is unaffected — the falsifier it was testing is whether the
> **ranking** moves, and it did not.

**If the learner branch opens it is a small, already-justified set**: the current `glm`, a
deterministic growing-basis spline, and the frozen `boost` candidate. Selection is on the component
losses, the `H₂`/`H₃` losses and predeclared remainder diagnostics — **never on coverage**. The
winner is serialised with its resolved parameters, seeds and thread settings, and the simplest
learner that closes the gap wins ties. A boosted learner additionally clears the direct stability
check or pins `reduced_crossfit="nested"`, for [the reason below](#why-the-oracle-comes-before-the-learner-and-why-boost-is-not-the-first-candidate).

**WP3 — the frozen confirmatory study.** E4's preregistration and E5's dispatch, run once, in that
order, after WP2 resolves and never before.

- the estimator and every learner **frozen before any result is generated**, on fresh seeds, two
  independent batches, both off-diagonal cells, `n = 600 / 1,200 / 2,400`;
- the primary design **predeclared and bound-inactive by construction**, with the bound-active
  stress cell reported separately — never as a post-hoc stratification of a mixed cell, which is
  [stop-ship 15](#stop-ship);
- zero score-identity failures and no more than the predeclared **2%** invalid-fit budget;
- a negligible or clearly vanishing `√n R_remaining`, an `se`/empirical-`sd` ratio inside its
  predeclared band, and batchwise coverage compatible with `0.95`.

**One of those is a proposed *tightening* of a frozen clause and is flagged rather than applied
here.** Gate 1's clause 1.5 currently reads `se ratio` in **`[0.90, 1.10]`**, which `q-drift`'s
`0.903` passes while the interval is ten per cent short of the spread it covers — a band wide enough
to pass a calibration this page has just spent a section on. `[0.95, 1.05]` is the band to commit,
and the place to commit it is [§5 of the validation
plan](drtmle/validation-plan.md#the-decision-rules-frozen-before-the-dispatch) **in E4's
preregistration commit, before E5 is dispatched** — the rules may move before the final run with a
written reason and not after it, and tightening is the direction that cannot buy a pass. Until that
commit exists the frozen band is still `[0.90, 1.10]`; this paragraph is a plan, not a rule.

**If the remainder gate passes and the `se` ratio still fails, that — and only that — is when the
calibration becomes its own question**, and it is E3's narrow form: paired pooled against nested
reductions, `‖Δ_k‖` measured directly, and a full-pipeline bootstrap as a *diagnostic*. Ship nested
reductions if they resolve it; otherwise derive and validate an estimator-specific correction. **A
blanket `se` multiplier is refused** — [see below](#what-this-piece-refuses-by-name).

##### The `se` shortfall is a symptom, and what withdrawing it costs

**The claim being withdrawn**, in the words it was made in, at
[link 4](#what-is-still-open), [the C3c reading](#what-c3c-measured) and
[stop-ship](#stop-ship): *`σ²ₙ` is the empirical variance of the estimated curve and treats the
reduced regressions as known, so their estimation error is in `ψ̂`'s spread and not in the variance
estimate.* Read as a description of the formula it is true and empty; read as this page read it — a
**second mechanism with a repair of its own**, worth about half the coverage gap — it is
unsupported, and three things say so.

*One, the formula is the theorem's.* Theorem 1's own last line is
`σ̂²_n = Pₙ[D*(Q*_n, g*_n) − D_A(Q̄*_{n,r}, g*_n) − D_Y(Q̄*_n, g*_{1,n,r}, g*_{2,n,r})]²` — the
uncentred second moment of the estimated corrected curve, [transcribed in the
concordance](drtmle/theorem-concordance.md) and pinned to the reported interval by
`test_theorem_drtmle.py::TestTheReportedVarianceIsTheorem1s`, which also measures that the
uncentred and centred forms differ by exactly `(Pₙ D)²`. The concordance's object table records
`drtmle`'s covariance block as the same object. **There is no missing term to name**: the
reductions' estimation error
enters the theorem through `R_{Q,n}` and `R_{g,n}`, which the theorem *assumes away* and which
item 13 measures as not negligible. The plug-in variance is first-order valid exactly when that
assumption holds, so a failing `se` ratio and a failing remainder are one premise failing twice, not
two defects.

*Two, the direction is not what an omitted term predicts.* A variance that omits a non-negative
component understates — in every cell, at every size. `g-drift`'s `se ratio` is **`1.157` and
`1.156`**: the reported `se` is 16% *larger* than the spread it is covering, in the same study, at
the same size, in both seed batches, against `q-drift`'s `0.903`. Strictly that does not prove the
term absent; what it proves is that something **larger and of the opposite sign** is in the column,
so the column is not a reading of an omitted variance term and the mechanism cannot be inferred from
it. One sign in each direction is what a *mis-estimated curve* looks like.

*Three, the split it licensed is order-dependent.* Model the interval as Wald at a standardised bias
`b` and an `se` ratio `r`, so coverage is `Φ(1.96r − b) − Φ(−1.96r − b)`. At `q-drift`, `n = 2,400`:
`sd = 0.0443`, `se = 0.0400`, `√n bias = 1.44`, hence `r = 0.903` and `b = 0.664`.

| counterfactual | model | gain |
| --- | --- | --- |
| as fitted | `0.8582` (observed `0.844`) | — |
| `se` correctly sized, bias kept | **`0.8982`** | `+0.040` |
| bias removed, `se` kept | **`0.9232`** | `+0.065` |
| both | `0.9500` | — |

The `0.8982` **is** the page's own `0.898`, reproduced to four places, so this is that arithmetic
rather than a rival one. Run the other way it says `+0.065` then `+0.027`. The two gains sum to
`0.105` against a gap of `0.092`: **they are not additive, so there is no split to report** — and
under either ordering the bias is the larger term. The critique's independent recomputation on the
495 valid `q-drift` fits agrees on the ordering and is more lopsided still, `0.885` for rescaling
against `0.927` for recentring.

**What this costs, said plainly.** No gate verdict moves: clause 1.5 failed in `g-drift` and still
does, clause 1.6 failed on coverage and still does, and `q-drift`'s `0.903` was always *inside*
`[0.90, 1.10]`. What moves is the *plan* — E3 stops being unconditional — and one sentence of
description in three documents. A retraction that moved a verdict would be a retraction doing work
it is not entitled to do, which is
[the rule this page already applies to `cancel`](drtmle/coverage-study.md).

> **On the artefact-level numbers quoted from the critique.** Its re-reading of the 3,000 C3c fit
> rows and the E2 rows — the `0.274` minimum margin, the `0/500` clipping, the `7/500` bound-active
> exit, the `0.885`/`0.927` counterfactuals — **cannot be reverified in the sandbox this page is
> written from**: Actions artefacts are served from `*.blob.core.windows.net`, which the proxy
> refuses (`CONNECT tunnel failed, response 403`), which is [why the manifest exists and why it
> carries digests](drtmle/study-manifest.md). They are cited as the critique's readings and each is
> corroborated by a published column that *is* in this repository — `min gr1 margin` `0.117`–`0.343`,
> `worst clip share` `0.0000` in eleven of twelve cell-runs, the Tier-2 bound-active rate of `2%` at
> that cell and size, and the `0.898` above. Nothing on this page turns on a number that exists only
> in the critique.

##### Why the oracle comes before the learner, and why `boost` is not the first candidate

**E2 is cheap, and that is most of the argument for putting it first — but it is not free, and an
earlier revision of this paragraph said three things that are not true.** They are corrected here
rather than deleted, because each was load-bearing for "nearly free".

*Withdrawn: "Tier 2's conditioning columns are injected analytic functions, so they are known in
closed form."* Tier 2 **fits** both primaries — an oversmoothed additive kernel regression and a
subset GLM (`benchmarks/drtmle_tier2.py`) — so only `branch_products`' `truth_*` coordinate is
analytic. That sentence describes Tier 1. And it fails at Tier 1 too once the alternation has moved
once: `refit` is handed the *targeted* pair, and `Q̄* = Q̄⁰ + ε·H` with a fitted `ε`.

*Withdrawn: "the existing coarse/fine bin pair is the discretisation check to reuse for the oracle's
own accuracy."* That pair is a **movement** between two rungs of a refinement, which is the
statistic [withdrawn above](#what-e1-landed-and-what-e1b-withdrew) for the quadrature ladder and now
for the branches themselves. E2 must not inherit a retracted statistic as its fidelity gate — which
is the whole reason its gate is specified below rather than assumed.

*Corrected: what the object actually is.* `conditional_mean`'s docstring is quoted accurately, but
the object E2 needs is **univariate** — `reduced.py` states it: *"Each is univariate: a regression
on one column, that column being the other nuisance's out-of-fold prediction."* The two-scalar
conditioning in `branch_products` belongs to appendix A/B's `0n` limits, a different object. So no
production call site computes what E2 needs, and the rows-per-cell coupling that makes a finer bin
count worse in two dimensions (576 cells, then 2,304) largely goes away in one.

**What is true, and it still puts E2 first.** The construction needs no change under `src/`:
`ReductionSpec.refit` is the seam, `CompanionEstimates` supplies every fold's nuisance at rows
declared before the fit ([item 24](#what-is-still-open)), `A` and `Y` are integrated in closed form by
`quadrature_frame`, and `tests/unit/test_oracle_reductions.py` already runs this override on the
exact law. What it costs beyond that is a companion grid and a fidelity gate — cheaper than a
learner sweep.

##### What E2 measured, and why it did not branch

**It ran.** One dispatch of `drtmle-reference.yml` at tier 2 and at the workflow's own defaults with
**no inputs passed**, on `main` at `6f3aeb3` — run `31042558057`, four jobs, 304 fits, about 87
minutes of runner time, [manifested](drtmle/study-manifest.md#e2-what-was-run) with digests. The
merge that made it dispatchable was the only thing it had been waiting on, and the previous
revision's `404` was that and nothing else.

**And it returned a verdict in one cell of four.**

| cell | `n` | gates | `ate` paired `d` | margin | verdict |
| --- | --- | --- | --- | --- | --- |
| `q-drift` | 600 | **fail** | `-1.3623` `[-1.6564, -1.0821]` | `±0.3850` | `unresolved` |
| `q-drift` | 2,400 | **fail** | `-1.0491` `[-1.1667, -0.9096]` | `±0.2899` | `unresolved` |
| `g-drift` | 600 | pass | `-1.7893` `[-2.2739, -1.3569]` | `±0.8997` | **`moved`** |
| `g-drift` | 2,400 | **fail** | `-1.6570` `[-2.1772, -1.1713]` | `±1.0203` | `unresolved` |

**Gate C passes everywhere**, by a factor of two to eight — the pilot sized `--reference-points`
correctly and that half came out as designed. **Gate B is what fails**, on one clause, three times:
*no other rung may be strictly better than the shipped reference*. `spline(8)` beats `spline(16)` on
`gr2` at `q-drift` 600, on `gr1` **and** `gr2` at `q-drift` 2,400, and on `qr` at `g-drift` 2,400 —
and on nothing at `g-drift` 600. `g-drift` at 2,400 fails a second clause too: the negative control
`bins(8)` is **not rejected** on `gr2`, so there the gate has no teeth.

**That pattern is [§8's own falsifier](drtmle/validation-plan.md#8-the-reference-comparison-piece-e2),
fired**, and it is worth quoting because it was written before the numbers: *"a gate-B ordering that
disagrees between the reductions — the shipped rung best on one and beaten on another — says the
reference's resolution is not one choice, and the repair is a per-regression resolution rather than a
verdict."* So the finding is about the *reference*, and E2's question — whether item 13's failure is
a learner failure at all — is not answered.

**Two revisions of this page then read that falsifier at a precision the table does not carry, and
the correction is worth more than the sentence it repairs.** They said *which rung wins differs by
cell **and** by reduced regression*. At the gate's own reading — a bootstrap interval over draws,
which is what `gate_verdict` tests and what the table prints — **no cell contains a disagreement
between its reductions**. Every interval that excludes zero within a cell points the same way:
`q-drift` `600` has `gr2` at `-2.855e-05 [-4.72e-05, -1.21e-05]` and nothing favouring the shipped
rung; `q-drift` `2,400` has `gr1` and `gr2` favouring `spline(8)` and `qr`'s interval straddling
zero; `g-drift` `2,400` has `qr` favouring `spline(8)` and the other two straddling. **What flips is
the cell**: at `g-drift` `600` the shipped rung is significantly the better one, `gr1` at
`+3.132e-05 [+5.19e-06, +6.44e-05]`. So the licensed statement is *one rung does not serve both
cells* — the per-regression pattern is real in the point estimates and is not resolved at 32 draws.
The repair changes accordingly: [E2R](#the-reboot-and-the-three-work-packages) selects per
`(cell, size)` **at least**, and per regression as a *procedure* rather than as a finding.

**What the run does and does not license, stated apart.** Nothing reads `equivalent`, so **candidate
1 is not dead and the learner road is not shut**. One cell reads `moved` with its gates passing, in
the direction candidate 1 predicts and with both arm means moving the same way rather than a contrast
cancelling something, so **candidate 1 is alive on one cell of four** — and one cell is not the
study, since §8's rule is per cell and size with no combination across them. The diagnosis does
**not** move to E3 on this run.

**The unread differences are large, and that sentence stops there on purpose.** The reference takes
`q-drift`'s column from `+1.1594` to `+0.1114` at `n = 2,400`, and all twelve rows move one way by
several times their own margins. Had those gates passed it would be candidate 1 alive by a wide
margin. They did not, and the exact-law measurement below is why the temptation is refused: a
two-bin reference is wrong about `qr` by more than half its own magnitude while its `ψ` sits inside a
twentieth of a standard error of an exact reference's. **A large paired difference at a reference
another resolution beats is a large difference about the wrong reference.**

**The next action is a repaired reference and not a larger study**, and it is cheap: the same design
with the rung **selected against a measured ranking** rather than shipped, on blocks that separate
selecting from certifying, with two composite gates on the covariates the estimator actually
consumes, and with the negative control confirmed rejected on `gr2` at the larger size first. That
is [E2R](#the-reboot-and-the-three-work-packages), it is a change to the *reference* and an addition
to the gates, which §8 permits before a dispatch with a written reason; the rule the comparison is
judged by does not move, and no constant in it was touched after a number existed.

**The three gates, and the point of each is what it is blind to.**

| gate | what it bounds | where it lives | blind to |
| --- | --- | --- | --- |
| **A** — the exact-law control | the **construction**: the weights, `qr`'s `\| A = a` mask, the arm columns, the recomputation at the current targeted pair | `tests/unit/test_reference_exact_law.py`, so it runs in CI rather than in a dispatch | the smoother and the quadrature, neither of which exists on a discrete law |
| **B** — the held-out weighted risk | the **smoothing bias**, by ranking candidates fitted on the reference block and scored on an independent finer one | `benchmarks/drtmle_reference_study.py`, per reduced regression, arm and fold | anything about the fit; it is a statement about three regressions |
| **C** — the randomisation budget | the **finite point count**, by independent scrambles of the reference block | the same module, on a declared subset of draws | the smoothing bias, which every scramble shares |

**Gate A is the one that turned out to have the most teeth, and it is worth saying why it works at
all.** The provider's own docstring said the object was owed a control "where the conditioning is
discrete and the answer is a finite sum", and the reason such a control is available is that under
`quadrature_frame` two of the three integrals are exact rather than approximate — `qr`'s `| A = a`
is a *weight* and not a subsample, and `gr1`'s target integrates to `g_0` through those same
weights. The discrete analogue of that frame is the exact law's own support at both arms, and a
`SaturatedCells` average over it **is** `_reduced`'s finite sum. The provider reproduces the law's
own reductions at the *exit* state of the alternation to `3e-14`. Two mutations were watched to
fail: extending the mask to `gr1` turns eight tests red, and flipping `qr`'s target sign turns two.

**And the trap the gates exist for is now exact rather than anecdotal.** The measurement below —
that a coarse reference lands a close `psi` — is now a statement on a law where the truth is
known: a two-bin reference is wrong about `qr` by more than **half its own magnitude** while its
`psi` sits inside a **twentieth of one standard error** of an exact reference's. A gate reading the
estimate would have passed it.

**Two things about gate B that a first draft would get wrong**, both recorded because each is a
claim. It is a **difference** of risks and never a ratio: a held-out risk is
`E_0[w(T − m)²] + E_0[w(m − m̂)²]`, and the first term is the irreducible variance of the target,
common to every candidate and able to dominate both — so a ratio sits near one whatever the
candidates are. And the three reductions are read **apart**, because `qr`'s target is a residual and
`gr1`'s is an indicator, so a mean over them is a number with no units and a gate on it is a gate on
whichever is larger.

**Gate C is E1b's device transported, and the one thing that does not transport is the cost.** There
the companion was inert to the fit, so eight replicates came off one fit; here the reference *enters*
the fit, so a scramble is a refit. Hence a declared subset of draws, its own flag, and a cost table
that prices a **fit** rather than a draw.

**Its own premise has been checked rather than assumed, and that is the one number this section
takes from a sandbox.** §8 names the falsifier: *a gate-C spread that does not fall when
`--reference-points` rises says the spread is not the quadrature, which would contradict the
unbiasedness the randomisation rests on*. A three-draw Tier-2 `q-drift` pilot at `n = 600` reads
`0.0637` at 4,096 points and `0.0315` at 8,192 — **roughly halved for a doubling**, which is the
shape a randomised quasi-Monte Carlo rule should have and is not the shape a smoothing bias would
have. Both pass the budget, whose allowance in that cell is `0.1420`, and the cheaper one does it at
`13.7` seconds a fit against `25.5` — so **E2's** default stayed at 4,096 with about a factor of two
in hand. E2R's is `8,192`, and the reason is not gate C but
[the ranking's own falsifier](#what-e2rs-instrument-landed), pulled and not fired. **The ratio is three draws' worth and carries no rate**, and the pilot's *comparison* columns
are not quoted anywhere: its primary verdict was `unresolved`, which is what three draws buy, and a
difference read under an `unresolved` verdict is the thing §8 refuses. Its rows were written to
git-ignored output and are gone, which is why nothing above depends on them beyond the flag's
sizing — [E1's own lesson](drtmle/study-manifest.md#the-study-manifest), applied to a knob rather
than to a study.

**The rule is frozen and it is [§8 of the validation plan](drtmle/validation-plan.md#8-the-reference-comparison-piece-e2).**
On the paired per-draw difference of `√n R_remaining`, against a margin of **a quarter of the `glm`
arm's own level** in that cell and size: `moved` if the bootstrap interval lies wholly outside the
band — candidate 1 alive, E2b fires; `equivalent` if wholly inside — candidate 1 dead, the learner
road shut, the diagnosis moves to E3; **`unresolved` otherwise, and it is a third verdict rather
than a weak `equivalent`**. A cell whose gates failed is `unresolved` however large its difference
is. Movement in *either* direction counts, because a reference that makes the remainder larger is
still a learner effect and E2's question is whether item 13's failure is a learner failure at all.
What is frozen is the **fraction**; the level it scales is the run's own control arm, which is a
measurement, and §8 says so rather than leaving a reader to notice.

**The record of what the instrument was, before the gates existed.**

**The instrument**, in `benchmarks/drtmle_reference.py`: a deterministic univariate
`SplineProjection` on the fitted index, injected through both hooks
(`ReferenceReductionDRTMLE`), with the `SaturatedCells` smoother the exact-law control needs and
the `EqualCountBins` negative control the gate must reject. `tests/unit/test_drtmle_reference.py`
pins the routing, the weights, linearity in the target, row-order invariance and the
points-per-parameter budget; three provider mutations — fitting off the reference block, routing
every row through one fold, and dropping `qr`'s `| A = a` — were each watched to fail.

**Two measurements from building it, and both change how the rest must be read.**

*A coarse reference lands a close estimate.* On one Tier-1 `q-drift` draw at `n = 600`, an
eight-bin reference reached `psi = +1.5108` against the spline's `+1.5129` and a truth of
`+1.5000` — **nearer the truth than the good one**, while being a far worse estimate of the
reduced functions and driving the alternation to its outer cap rather than to tolerance. So a
gate that reads the estimate would have passed the bad reference, and E2's branch cannot be read
off `√n R_remaining` until the *function* has been gated.

*Fold routing is invisible on Tier 1.* Both nuisances there are injected analytic functions of
`W`, so every outer fold's companion copy is identical — the fold sums came back `1402.0213`
five times. Any routing test taken on Tier 1 passes against any routing whatever, which is the
same shape of blindness `CLAUDE.md` records for a saturated learner and a redundant column. The
routing test therefore runs on fitted primaries.

**What was owed before any paired number could be read** was the three fidelity gates, the
precommitted equivalence margin and reference-uncertainty budget frozen in their own commit, and
the dispatch. All four landed, the run has happened, and **the gates are what stopped the paired
numbers being read** in three cells of four — which is the instrument working rather than the run
failing, and is the whole return on specifying them before any number existed. What is owed now is a
reference whose resolution is **selected and then certified by a block that did not select it**,
which is [E2R](#the-reboot-and-the-three-work-packages).

**And it is the only instrument that can return a *negative* answer worth having.** A faster
learner that fails leaves "try a faster one still"; an oracle that fails says candidate 1 is not
where the remainder is, which is the sentence that redirects the piece.

**The obvious flexible candidate is `boost`, and this page refuses to reach for it first — on its
own argument rather than on caution.** [A1b's stability condition](drtmle/theorem-concordance.md)
reads, of `‖Δ_k‖ = o_p(1)`: *free for a fixed-basis smoother; **not** free for anything selecting a
split, a bandwidth or a candidate from the data, **which includes the default `boost`***. So
putting LightGBM in the reductions forfeits the only argument the **pooled** construction has, and
buys a coverage number whose cross-fitting is then unsupported — trading one `unverified` row for
another and calling it progress. If E2b promotes a data-adaptive learner it must either extend
A1b's entropy argument or pin `reduced_crossfit="nested"`, and it must say which. A deterministic
growing basis (`k_n → ∞`, `k_n/n → 0`) is the candidate that needs neither, and the package already
ships the spline expansion to build it from.

##### What this piece refuses, by name

- **A larger `n` on the same configuration.** Named first because it is the intuitive next step and
  the measurements rule it out.
- **A bootstrap or sandwich variance to widen the interval, and a blanket `se` multiplier most of
  all.** The refusal stands and **its reason has changed**: the previous revision said the `se`
  ratio has a *known* mechanism that E3 addresses, and
  [it does not](#the-se-shortfall-is-a-symptom-and-what-withdrawing-it-costs). The reason is now
  the stronger one — the calibration and the remainder are **the same unestablished premise**, so
  an independent variance inflation would mask the mechanism rather than repair it, and it cannot
  touch the bias, which is the larger of the two terms under either ordering. A full-pipeline
  bootstrap is admissible **as a diagnostic** in WP3's conditional branch, and only there.
- **Tuning any learner on the final coverage**, and **selecting the reduction learner on anything
  but reduced-function loss and the theorem products**. C3c's own value is that its rules were
  frozen before its numbers existed.
- **Piece D — bivariate reductions and the wider treatment scope — stays deferred behind WP3, and
  it is not a fallback *inside* this release.** A second reduction *construction* is not a response
  to an unvalidated reduction *learner*; and if no feasible univariate learner reaches the reference
  at `n = 2,400`, that is [a stop result](#the-branch-e2r-decides) and the bivariate estimator is a
  separately specified algorithm project, not the rescue for this one.
- **Conditioning any primary number on the post-fit bound label.** E4 separates the designs; it
  does not stratify a result after the fact, which is [stop-ship 15](#stop-ship).
- **Citing E2's three `unresolved` cells as motivation for a production change.** Their differences
  are large and they are about a reference another rung beats; the piece may cite them as the reason
  E2R exists and for nothing else. **The same bound applies to E2R itself**: an `unresolved` decision
  run ends the reduction road as evidence, rather than earning a third dispatch.

##### What E1 landed, and what E1b withdrew

The remainder instrument, and it **closes no numbered item** — item 13 is a rate and closes at E5.
What it landed is the *mechanism*, and that stands. What it also shipped was two readings taken off
that mechanism, and **both are withdrawn**; they are stated here rather than deleted, because a
page that quietly stops saying something is a page a reader cannot audit.

*Withdrawn: "a nested ladder bounds the grid's own error at one to three per cent of the column's
spread."* `delta` is the paired movement of the column between a rung and the next coarser one, and
a successive difference is **not** an error bound without monotonicity or an applicable convergence
result — which this integrand does not have, as E1's own module docstring said of the Tier-2 kernel
cutoff before drawing the opposite conclusion. Measured on a piecewise-smooth integrand at this
ladder's own geometry, the finest rung's `delta` ran **`0.000102` against an actual error of
`0.000427`**, four times smaller; on a smooth one it also undershot, and two rungs earlier it
overshot by three orders. It is a *stability* statistic in both directions and an error estimate in
neither.

*Withdrawn: "97% to 99% of the across-draw variance of `√n R_remaining` was the evaluation draw."*
`var removed = 1 − s²_grid/s²_draw` is not that share. Writing `R_grid = X + e_grid` and
`R_draw = X + e_draw`, the two arms come off the **same fits** — the companion is inert, so this is
a stronger pairing than the page claimed and `Var(X)` cancels exactly — and `E[e_draw | fit] = 0`
kills its covariance. What does **not** cancel is `2·Cov(X, e_grid)`: a *fixed* grid's error is a
deterministic function of the fitted curve. So the ratio estimates
`[Var(e_draw) − Var(e_grid) − 2·Cov(X, e_grid)] / Var(R_draw)`, and the last two terms were bounded
only by the ladder, which the paragraph above says bounds nothing.

*Withdrawn, and it is not E1's: `branch err` was the same statistic one level down, and the first
two retractions left it standing.* `branch_error` is the binned limits' movement between
`BIN_COUNTS = (12, 24)`, and `benchmarks/drtmle_remainder.py` reported it as "the error of the
binned limits" while `validation-plan.md` §5 gated on it — *"a branch smaller than that error is
reported as not resolvable"*. It is a successive difference between two rungs of a refinement, so
it is `delta` with bins in place of Sobol points, and everything the first paragraph above says
applies verbatim. The column is now `branch move` and the count is `branches settled`.

Three things follow and each is written where it is read. **The suppression stays**, because one
direction of the inference does hold: a branch moving more than its own magnitude is an instrument
visibly still moving. The converse does not — `test_settling_is_not_sufficient` constructs a target
with one period per fine bin, where the two counts agree to `2e-15` and the residual is the *whole*
of the target — so settling is necessary, not sufficient, and a settled branch's error is
**unestablished** rather than small. **`cancel` at `1.99x` becomes unread rather than failed**, and
[clause 1.4's verdict does not move](drtmle/coverage-study.md#the-gates-read-out-clause-by-clause),
because `√n R_rem` carries it alone. **And randomisation does not rescue this one**, which is the
asymmetry that makes it E2's problem rather than a second E1b: an independent scramble makes a
*quadrature* error mean-zero, and every randomisation of a bin count shares the same partition
rule, so the across-scramble spread is orthogonal to a bias in the bins.

*Not withdrawn, and worth separating from the three above.* The draw-count objection applies to the
**equivocal** cells and not to the headline ones: a conservative independent-`F` interval puts
`0.974` at sixteen draws inside `[0.938, 0.989]` and at eight inside `[0.902, 0.993]`, while the
`−0.069` cell spans `[−3.05, 0.72]`. The record's "a variance ratio at eight draws has a standard
error near `0.5`" was true of the cell it was written for and was then read as if it applied
everywhere. What must go regardless is "**reproduces it exactly**": two estimates with those
intervals agreeing at the third decimal is a coincidence and not a reproduction.

*What survives is two measured spreads and their ratio, without the attribution.* At `n = 2,400`
the across-draw spread of the column was `2.05` under C3c's rule and `0.33` under the grid, in Tier
1's `q-drift`. Those are two standard deviations of a recorded column; they need no identification
argument, and they are why the deterministic rule is worth having at all. [What the ladder
measured](drtmle/investigation-log.md#what-the-e1-ladder-measured) carries the tables with the
retraction attached to each.

- **`P₀D̂` by deterministic integration**, `benchmarks/drtmle_remainder.quadrature_frame`, and it is
  a **reduction of the problem rather than a bigger draw** — which is the piece's one real design
  decision. The corrected curve is *affine in `Y` given `(A, W)`* and reads `A` *only through the
  indicator*: `reduced_correction_parts` builds `D*_g = (Q_r/g)(1_a − g)`, which has no `Y` in it,
  and `D*_Q = 1_a·(g_{r,2}/g_{r,1})(Y − Q̄*)`, which is affine in it. So
  `P₀D̂ = E_W[Σ_a g₀(a|W)·D̂(W, a, Q̄₀(a,W))]` is an **identity**: the sum over `A` is finite, the
  integral over `Y` is closed form, and what is left is a quadrature in `W` alone — taken on the
  Sobol rule `truth()` already integrates with, so `ψ₀` and `P₀D̂` come off one grid;
- **and `ψ₀` moving onto that grid is most of what the rule buys**, which was not obvious and is
  worth the sentence. Substituting the curve's centring, `R_remaining = E₀[D̂ᵘ − Q̄₀] − PₙD̂`, and
  the equality holds *only if both expectations are one integral* — under which every term of the
  integrand is a product of two nuisance errors rather than an `O(1)` quantity. Take `ψ₀` from the
  default rule instead and two grids are differenced, and the `O(1)` part of the error survives.
  `DGP.quadrature` exists for that and is the piece's only change under `src/`;
- **a ladder, and it is one fit per draw.** The grids are nested prefixes and the companion
  interleaves the arms, so the first `2k` rows *are* the grid at `k` points and a rung is a `Window`
  slice — which makes the movement between two rungs the discretisation rather than a difference
  between two fits that would have to be argued bit-identical. Pinned against an actual refit, bit
  for bit on every column. **What that movement is evidence of is where E1 overreached**: it says
  the sequence settled, it does not bound where it settled, and E1b replaces the reading without
  deleting the ladder;
- **the full score row per fit**, `<stamp>-scores.jsonl` beside the replicates from one timestamp.
  `valid` and its two failure counts stay exactly as they are, because gate 1's clauses 2 and 3
  read them apart; the rows are *in addition*. A count cannot say which equation missed, by what
  ratio, or whether it started large and was driven down — and that last is what E3 replays the 99
  invalid fits with;
- **and `branch_error` is read by a table for the first time.** It has been on every replicate
  since C2 and was displayed nowhere, which is why C3c's `branches settled` falling to `192/250`
  arrived with nothing beside it. *What it is* has since been withdrawn — it is a movement between
  two bin counts, not an error — and the column is now `branch move`. That column is also where the sweep found the
  thing it was not run for: under the draw the binned limits put 2,000 rows into 576 cells, and in
  both `g-drift` rows of Tier 2 the two rules **disagree about `R_g`'s sign**. Four times the rows
  is what buys it, and it is a direct reading on clause 4's second half.

What it does **not** land, by name. **Any coverage claim, and any rate**: a quadrature's own error,
however it is measured, says nothing whatever about the remainder. **Any learner selection**, which
is E2's and E2b's. And **no status moves in the concordance**: an instrument landing is not a
condition being met, and neither is a sharper instrument.

**A reading it declines to make in either direction**, because the temptation runs both ways here.
E1 does not say whether the sharper column resolves C3c's decline. What it can say is arithmetic:
at 250 draws the Monte Carlo error on `√n R_remaining` would be several-fold smaller under the grid
— sixfold and eightfold at Tier 1, from the two measured spreads and nothing else — and whether
that separates a decline depends on the decline, which E5 measures. Both directions are readings of
a rate and neither is this piece's.

**One thing it corrects in the record**, and it is [lesson
19](drtmle/investigation-log.md#what-the-sizings-got-wrong): the `sd(D)/√m ≈ 0.023` this page has
quoted for three revisions was computed from an *assumed* `sd(D) ≈ 1.0` and is `1.4x` to `1.8x` too
small. **The corollary that lesson drew is itself wrong**, and E1b is the correction: differencing
two measured spreads is not the decomposition, because only one of the two error terms is
mean-zero given the fit.

##### What E1b measures

**The same question, with the statistic changed so that it answers it.** One mechanism does all of
it: an **independent scramble per replicate** rather than one fixed scrambled Sobol sequence.

- **It makes the grid's error mean-zero.** A randomised quasi-Monte Carlo rule is unbiased at every
  point count — the randomisation is over the scramble, not over the points — so `E[e_grid | fit]`
  is zero and `Cov(X, e_grid)` with it. The differenced share becomes identified rather than
  bounded, and the sign of what is left is known.
- **It makes the error estimable from replication rather than from refinement.** Independent
  scrambles at one grid give a standard error directly, assuming nothing about a rate; nested
  prefixes *within* each scramble keep the ladder, now labelled what it is.
- **It gives the draw's share without any differencing at all.** Several independent i.i.d.
  companions per fit measure `Var(e_draw | fit)` conditionally, so `draw share` is a ratio of two
  things both estimated rather than a subtraction — and `var removed` stays beside it, so two
  routes to one number can visibly disagree.
- **It costs one fit per draw and no refits.** The companion is inert to the fit, so every
  scramble and every i.i.d. replicate is a row block of one stacked companion, read through a
  `Window` where E1 read a prefix.
- **And it is strictly better for E5**, which is the part that was not obvious when the fixed grid
  was chosen. E1 traded the draw's averaging-down noise for a bias no replicate count removes. A
  randomised scramble gives up nothing to get that back: the error is noise again, it is far
  smaller than the draw's, and it averages down over the study.

**What it measured**, in one table — the finest rung of each ladder, 32 draws a cell and eight
replicates of each rule per fit, `share` being the fraction of a one-replicate study's across-draw
variance that the rule accounts for:

| tier | cell | `n` | `share`, grid | `share`, i.i.d. draw at `m = 2,000` |
| --- | --- | --- | --- | --- |
| 1 | `q-drift` | 600 | `0.000` `[0.00, 0.00]` | `0.821` `[0.73, 0.93]` |
| 1 | `q-drift` | 2,400 | `0.000` `[0.00, 0.00]` | `1.003` `[0.95, 1.05]` |
| 1 | `g-drift` | 2,400 | `0.001` `[0.00, 0.00]` | `1.008` `[0.96, 1.06]` |
| 2 | `q-drift` | 600 | `0.002` `[0.00, 0.00]` | `0.723` `[0.60, 0.89]` |
| 2 | `q-drift` | 2,400 | `0.081` `[0.02, 0.11]` | `0.991` `[0.94, 1.05]` |
| 2 | `g-drift` | 2,400 | `0.051` `[0.02, 0.10]` | `0.790` `[0.69, 0.91]` |

[The full eight cells](drtmle/investigation-log.md#what-the-e1b-dispatch-measured), including the
two `n = 600` `g-drift` rows. A share reaching past one is a **draw count**, not a finding: `Var(X)`
is a difference, and where a rule's error dominates by an order it is not resolvable at 32 draws.

**The retraction was about the statistic and not about the size**, and this is where that becomes
visible: E1's Tier-1 `0.974` and `0.987` sit close to these `1.003` and `1.008`, while its
`−0.069` and `0.204` at `n = 600` are `0.723` and `0.808` here. Its column happened to be near the
right answer in the cells it was quoted from and was uninformative in half the others, which is
what an unidentified statistic without an interval looks like when it is lucky.

Its numbers come from a dispatched run with retained artefacts, identified in [the study
manifest](drtmle/study-manifest.md#e1b-what-was-run), because E1's came from a sandbox whose output
was git-ignored and is gone.

#### D. Widen the scope to what the sources derive

**Closes the two candidates in item 10.** Everything else in that item is a refusal with a reason,
not a gap. Both candidates are gated on reading rather than on writing, which is why they sit
behind **A1a** — the reading — rather than beside it.

- **`reduction="bivariate"`.** van der Laan (2014)'s original single bivariate
  `gr(a | w) = P(A = a | Q̄̂(a, W), ĝ(a|W))` in place of the `gr1`/`gr2` pair, with equation (10′)
  in place of (10). It is derived in the sources and was in scope; it was cut because it is a
  different extra equation on a two-column design rather than a wider loop over the first, and
  nothing was waiting on it. **"Derived" is not "transcription"**, and calling it that before the
  score, the correction term and the targeting step are mapped would repeat exactly the mistake
  items 1 and 2 exist to fix — the more so now that item 21 shows what a transcription can hide.
  **Its theorem is not in hand**: the 2023 article reproduces the equations (6)–(10) and refers
  the regularity conditions to van der Laan (2014) Theorem 3, which nothing here has read, so
  the missing statement, its assumptions, its expansion, its influence function and its remainder
  decomposition are all outstanding — [the concordance's
  §10](drtmle/theorem-concordance.md#10-the-bivariate-construction) records the status. It gets
  its own reduced object, its own submodel and its own fixtures rather than being folded into
  `ReducedSet`'s array schema — two reductions whose estimating equations differ structurally
  should not share a container — and it is worth a side-by-side run against the univariate pair,
  since which of the two is better behaved on a real fit is not something either source settles.
  One detail to carry over from the R source rather than rediscover: its bivariate branch of
  `eval_Dstar_Q` is `1{A=a}/grn2 · (grn2 − g)/g · (Y − Q)` and the `g` there is the **initial**
  mechanism, not the targeted one — `drtmle.R` passes `gn = gn` into that call in both the loop
  and the covariance block. On the univariate branch the argument is unused, so this is a
  difference that only appears when the bivariate reduction is written.
- **A multi-valued treatment.** The obvious reading of the source says this is already licensed —
  `drtmle(a0 = c(0, 1, 2))` reports treatment-specific means at `K` arms and the software paper
  works an example and builds a covariance matrix over the means and their contrasts, the
  estimating equations are written with a free `a`, and nothing in them has a two-arm step. What
  is missing is the derivation: van der Laan (2014) states its problem for a "subsequently
  assigned **binary** treatment", and no theorem read here covers `K` arms. An implementation that
  accepts an argument is not a proof that the argument is licensed. The questions such a reading
  has to answer are the right list: are the arm-level means targeted jointly or one at a time; is
  the targeted mechanism still on the simplex; are the armwise tilts variation-independent; what
  is the *joint* corrected curve and hence the covariance a contrast needs; is positivity
  arm-specific; and does the theorem hold `K` fixed.

  **The simplex question is an unasked question, not a known defect**, and the previous revision
  of this page called it one. The per-arm mechanism tilts do not renormalise, so the targeted
  `g*(·|W)` at `K` arms is not a distribution over the arms — but an algorithm may use
  arm-indexed nuisance updates as working objects for separate estimating equations without
  claiming that they jointly define one conditional treatment distribution, and whether the
  theorem requires a single valid joint `g*` or only arm-specific functions satisfying the stated
  equations is exactly what is unread. `drtmle` does not renormalise at **two** arms either:
  `fluctuateG` tilts each arm's mechanism in its own one-column `glm`, so `g*(1|W) + g*(0|W)` need
  not be one there. That is not a licence — it is the same unasked question, already live in the
  reference implementation, and the honest description of the binary path meanwhile is *an
  arm-specific parameterisation whose joint-distribution interpretation must be checked against
  the theorem*. For a new multi-arm implementation prefer a **simplex-preserving multinomial
  fluctuation** unless the theorem explicitly licenses independent armwise updates; contrasts need
  a coherent joint covariance construction whether or not each arm mean is targeted separately.
  `solve_mechanism` has no multi-arm tilt today, since `ipsi` declares
  `requires_binary_treatment` and has never needed one.

**The order to work in**, revised again, and it follows from what blocks what rather than from
effort. Piece **0** was first and has landed, and so now have **B1a**, **A1a**, **B1b** and
**B2a**; what is left is:

1. ~~**B1a**~~ — landed. It was first because it is the only piece that changes a number every
   other piece reads, because it is valid under every convention A1a might select, and
   because it is a patch plus its tests rather than a study. [What it
   shipped](#what-b1a-landed).
2. ~~**A1a**~~ — landed. Items 21 and 22 closed on the source — the sign in the implementation's
   favour, the update order because the paper prescribes a fixed point rather than a route — and
   item 1 closed here, along with the concordance's `evidence` column, its assumption matrix,
   §7's finding for B1b, and the decomposition test the `test_influence_gateaux*` modules could
   not supply. None of it needed a document that is not in hand and none of it needed another
   implementation.
3. ~~**B1b**~~ — landed. A1a had said which mechanism the theorem's `D_g` is evaluated at ([the
   concordance's §7](drtmle/theorem-concordance.md#7-truncation-is-not-in-the-theorems-algorithm)),
   and since the theorem clips nothing there was no implementation whose convention could settle
   this and no document left to wait for — so it was a design decision against a stated bar rather
   than a reading, taken against a fitted prototype rather than against the candidate table.
   [What it shipped](#what-b1b-landed).
4. ~~**B2a**~~ — landed, in two passes. The dispatch could not measure what [the validation
   plan's §4](drtmle/validation-plan.md#4-the-sweep-piece-b2) asks for until the script recorded
   it, and item 22's numerical half asks for a comparison against an update order that did not
   exist here. [What it shipped](#b2a--the-sweep-instrument). The second pass closed the three
   things the first left as prose: the oracle reduction is built where it exists and **item 24**
   says what it would take elsewhere, the update-order difference has a control and a rule frozen
   before the dispatch that judges it, and the branch that let a mutation hide is gone.
5. ~~**B2b**~~ — landed. The exit distribution under the current rule was uncharacterised and now
   is not: it inverted, 87 of 96 fits reaching the tolerance where 2 did, and the sweep costs a
   seventh of what it did. `weak-overlap` was the reason to run this before the demonstration, and
   it turned out not to be a problem to route around — 0 of 24 score-check failures against 23 of
   24, on draws whose overlap is unchanged — so no refusal is predeclared. Items 12 and 19 closed;
   limitations 4 and 6 were rewritten from their own numbers rather than from the first sweep's.
   [What it measured](drtmle/investigation-log.md#what-the-b2b-dispatch-measured). The one thing it
   did not settle is item 22's numerical half on `weak-overlap`, where twelve draws resolve nothing
   in either direction.
6. ~~**The [contract](#the-supported-contract-and-item-25)**~~ — landed, prose and the cheapest
   thing on the list. It was ordered here rather than "anywhere" because it is what C's numbers are
   read under, and a scope inferred from a completed study is not a scope. Its per-fit witness came
   with C1 and **found its own second bullet overstated**, which is what having a witness is for.
7. ~~**A1b**~~ — waited on nothing and has landed. Its construction decision is now **frozen at
   the pooled construction**, which is what C's final dispatch needed — the argument closes on the
   entropy condition and the dispatch supports the stability one without showing it, and the thing
   that would reopen the decision is a `‖Δ_k‖` measurement rather than another sweep of this one.
   [What it measured](drtmle/investigation-log.md#what-the-a1b-dispatch-measured).
8. ~~**C1**~~ — landed: the harness, Tier 1, the workflow, the witness, and
   [the design note](drtmle/coverage-study.md) with its drift coefficients committed and verified.
   The **sizing was redone rather than inherited**, which is what this list asked for: C had been
   costed from a 43s `DRTMLE` fit measured before B1b, and the re-timing puts that same fit at
   **5.6s** — a factor of 7.7 — and a *Tier-1* fit at **1.2s**, since its primary nuisances are
   function evaluations rather than learner fits. So a 300-draw pilot is under an hour rather than a
   day. **The study
   design needed nothing added**, exactly as the last revision concluded: a review proposed two
   off-diagonal regimes, a known nonzero drift, three sizes, the remainder branches reported apart,
   coverage with `se` calibration and per-replicate results kept, and
   [§5](drtmle/validation-plan.md#5-the-controlled-study-piece-c) already specified every one.
9. **C2**, Tier 2 and item 13: prescribed-rate learners, and the fold-retained nuisance object
   `P₀D̂` needs. It re-times before it re-scopes, as C1 did — its nuisances are *fitted*, which is
   what the 43s figure was measuring, so C1's 1.2s does not transfer.
10. ~~**C3c**~~ — landed, and the order held: nothing entered it unfrozen, including the
    mixed-cell reporting decision and both batches' seeds, which were committed before the first
    fit. It answered item 3 by measuring it — the gap is there and large — gave item 13 its first
    number without closing it, and left four stop-ship clauses live on numbers rather than on the
    absence of them. [What it measured](#what-c3c-measured).
11. **E0 through E5**, [what C3c handed back](#e-what-c3c-handed-back): the record first, then the
    remainder instrument, then the **oracle** reduction that decides whether the learner is where
    the flat remainder lives, then the construction and the solver, then two designs where there
    was one, and only then a fresh dispatch. The ordering is the same rule the rest of this list
    is built on — a run whose inputs are not established measures a design nobody has checked, and
    C3a and C3b are what that cost twice. ~~**E0**~~ and ~~**E1**~~ have landed and ~~**E1b**~~ is the
    one between E1 and the oracle, and it has landed: the instrument's mechanism is settled and its
    residual error is measured, which is what had to be true before E2's oracle could be read as a
    statement about the reductions. [What E1 landed, and what E1b
    withdrew](#what-e1-landed-and-what-e1b-withdrew).

**Item 23 was outside that order**, like **D**: a small fix with an oracle already in the
repository, touching only the partial-guard path, which is why it did not wait on the theorem or on
anything else here. It has [landed](#closed-since-this-list-opened), and **D** is what is left
outside the order.

Then the applied stress tests, which are generalisation checks and not substitutes for C: a Super
Learner library, higher-dimensional and nonlinear processes, moderate near-positivity, binary as
well as continuous outcomes, fixed analysis weights, repeated cross-fitting, different
reduced-regression learners and a truncation grid, at sizes representative of use. Keep the
per-replicate results and not only the summary tables. A **fitted** weighted run belongs on that
list too: item 17 closed the *transport* on the exact law, which is the right instrument for an
identity about conditional expectations, and is not the same thing as having run one. `repeats=`
is off the list — item 18 ran it. **D** is independent of all of it and should not queue behind
any of it.

### Stop-ship

Any one of these blocks calling `DRTMLE` finished, and they are the four links restated as things
a reader could check rather than as claims. **Four** of them are about how the *evidence* is
described rather than about the code — 11, 14, 15 and the new **16** — and all four exist because a
claim that outruns its instrument is how this variant has gone wrong three times now.

**Two further stop conditions are not clauses here because they are outcomes of an experiment
rather than properties of a document**, and [the branch table](#the-branch-e2r-decides) is where
they live: a repaired reference that still cannot say whether the reductions are causal, and a
reference that repairs the remainder with no feasible learner reaching it at `n = 2,400`. Each ends
the release attempt with a recorded result rather than with a further dispatch.

**Four of them are now live on measurements rather than open for want of one.** Before
[C3c](#what-c3c-measured) this list was mostly a set of things a study might find; the study found
them. Clauses 4, 5, 6 and 7 are live, on 6,000 fits over two seed batches, and clauses 8 and 9 were
checked and are not. That is a better state than it sounds: a list of blockers with numbers
attached is a list somebody can work on.

**They do not all trace back to one quantity — and this revision moves clause 5 back towards the
one they mostly do.** Clause 6 — coverage — is the one the remainder drives. **Clause 5 is the
`se ratio`, and it no longer has a mechanism of its own**: the previous revision named the variance
estimator, `σ²ₙ` "treating the reduced regressions as known", and that is Theorem 1's own formula
rather than a defect
([the withdrawal](#the-se-shortfall-is-a-symptom-and-what-withdrawing-it-costs)). It is the same
unestablished premise as clause 6, read through the second moment, and `g-drift`'s `1.157` is why
it cannot be an omitted variance term. Clause 4's second half is `cancel` at `1.99x`, which is the
branch decomposition failing to *separate* the two remainder terms rather than either of them
failing to vanish. Clause 3 is 99 invalid fits, which is the solver. **Four clauses, three
mechanisms, and now two of the four share one** — which is why
[piece E](#the-reboot-and-the-three-work-packages) is a diagnostic, a branch and a study rather
than four parallel repairs. A single-cause reading would still send work to the wrong place; so
would inventing a cause per clause.

1. a correction term disagrees with Theorem 1 or the appendix — this was item 21 and it is
   **closed**: the appendices force the positive reading, which is the one implemented, and
   `tests/unit/test_theorem_drtmle.py` is what a reader checks that against rather than the
   §3.1 display;
2. the algorithm's fixed point is not shown to satisfy the theorem's three score conditions —
   item 22, whose *theoretical* half is closed (the paper's step 7 states its own exit as those
   three empirical means, so the order is not prescriptive) and whose numerical half is now
   [B2](#b2--the-sweep-on-the-corrected-implementation)'s: whether the paper's order and this one
   reach the same fixed point on real data, both orders run here against the same nuisances. It
   used to say the numerical half was the parity piece's, which was never right — a second
   implementation reaching a third fixed point would have answered a different question — and
   then A1's, which was right about the subject and wrong about the evidence: it is a second
   alternation over a sweep of draws, which is B2's dispatch and nothing A1a could share with.
   **The second alternation exists** — `DRTMLE(update_order="paper")`, landed with
   [B2a](#b2a--the-sweep-instrument) — and [B2b](#b2b--the-dispatch-and-what-it-decides) has read
   it over a sweep. **It stays on this list, and the reason is one clause.** Of [the rule frozen
   before the dispatch](drtmle/validation-plan.md#the-update-order-rule-frozen-before-the-dispatch),
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
   evidence and [the
   log](drtmle/investigation-log.md#the-same-rule-at-thirty-six-draws-and-why-the-two-readings-are-not-nested)
   records the two restatements a future revision could make **before** a further dispatch and not
   after this one;
3. a stored score and the term the curve carries are not the same functional of the same state —
   this was item 20, the one that was true and unnoticed, and it is **closed**:
   [B1a](#b1a--the-identity-and-safety-patch) made it reported and
   [B1b](#b1b--the-theorem-conforming-targeting-decision) made it false, by solving the score at
   the truncated tilt the curve reads. It stays on this list as a thing a reader can check rather
   than as an open item, and the check is `res.validation.correction_check()` on any fit —
   including one whose bound binds, which is the case the numbers used to come from;
4. a required score is not negligible under the predeclared validity rule — **[C3c](#what-c3c-measured)
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
    §8.5](drtmle/theorem-concordance.md#85-what-a1a-settled-and-the-reason-it-gave-was-wrong) says
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
    written to catch. `tests/unit/test_nested_reductions.py::TestADataIndependentPrimaryLearnerMakesTheTwoConstructionsAgree`
    is the corrected statement, asserted and kept as a mutation watched to **pass**;
15. any document reads a **bound-active** fit's numbers as evidence that Theorem 1's expansion
    covers it. B1b made the solved score and the reported curve one expression at one state, which
    is internal coherence; the theorem has one mechanism and it is untruncated
    ([§7](drtmle/theorem-concordance.md#7-truncation-is-not-in-the-theorems-algorithm)), so a fit
    on which `ĝ` or `g_{r,1}` sits against a bound is [item
    25](#the-supported-contract-and-item-25)'s third option — empirically supported, outside the
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
    a missing non-negative term looks like.

Note what is **not** on this list any more: a coverage gap over `TMLE` of at least 0.05. That is a
product judgment about whether the variant earns its cost, it has no theorem behind it, and it now
lives in [gate 2](drtmle/validation-plan.md#the-decision-rules-frozen-before-the-dispatch).

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
first: [before](drtmle/investigation-log.md#how-the-alternation-exits), 2 fits reached the
tolerance, 86 stalled and 8 ran out of rounds;
[after](drtmle/investigation-log.md#what-the-b2b-dispatch-measured), **87 reach the tolerance, 8
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
adopted one and four fixtures could not tell; **[B2b](#b2b--the-dispatch-and-what-it-decides) can,
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
two candidates, and they are [piece D](#d-widen-the-scope-to-what-the-sources-derive).

### Closed since this list opened

Kept rather than deleted, because the numbering is frozen: these items are cited by number from
`benchmarks/bench_drtmle.py`, `.github/workflows/drtmle-convergence.yml`,
`estimators/targeting.py` and `tests/unit/test_drtmle_fit.py`, so a closed item's number is not
available for reuse. Items 14, 16, 17 and 18 were **piece 0**, which is why its section is gone
from the list above: none of the four was research, all four were claims the package made that
were wider than the evidence behind them, and what they protected a user from was being told
something the fit had not earned while everything else here is open.

**11 and 20. The reported curve is centred wherever the truncation binds, and the two items were
always one failure.** The full diagnosis stays where it was written, in
[piece B2's section](#b2--the-sweep-on-the-corrected-implementation), because it is the evidence
B1b was chosen against; what closed it is [B1b](#b1b--the-theorem-conforming-targeting-decision).
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
[§9](drtmle/theorem-concordance.md#9-what-was-read-out-of-the-r-source-and-what-is-still-owed)
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
re-propose. **No R script will enter this repository and no R step will enter CI** — not for
`drtmle`, not for `tmle`, `tmle3` or `ctmle`, and "it would only be one file" does not reopen it.
`CLAUDE.md` carries the rule; this is the reasoning.

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
([lesson 9](drtmle/investigation-log.md#what-the-sizings-got-wrong)). What is not blind is a
comparison against the theorem's own terms at a **nonzero `Q_r`**, which is
`tests/unit/test_theorem_drtmle.py`.

And the empirical record points the same way. Item 20 sat in this piece for two revisions on the
reasoning that a divergence between two arrays is what a component-by-component comparison locates.
The reasoning was sound and the premise was false — there was no divergence between two arrays —
and what found it was thirty lines, one fit and **no R**: recomputing the recorded score from the
returned state in the same process. That is
[lesson 8](drtmle/investigation-log.md#what-the-sizings-got-wrong), and it is the second time this
page filed a finding behind a fixture that would not have produced it.

**What survives is the decomposition, and it moves to A1a.** "Compare components, not `psi` and
`se`" was always the right instruction — several differences cancel at `psi`, and `psi` is precisely
what all three empirical means being zero makes insensitive to the corrections. Only the thing each
component was compared *against* was wrong. Each is now checked against its derivation, and the
[concordance's object table](drtmle/theorem-concordance.md#13-the-object-concordance) gains an
`evidence` column naming, per object, the test that pins it; the rows still reading `TODO` there are
what is left of this item, restated as tests to write rather than as a fixture to import.

The [first review](drtmle/review.md)'s recommendation A3 — *"cross-language validation at the
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
[piece B2](#b2--the-sweep-on-the-corrected-implementation)'s and needs its evidence. Serialising
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

### A differential diagnostic against R, considered and refused

**A review of E2R's instrument asked for the R refusal to be narrowed, and the distinction it
draws is a real one that the [standing decision](#standing-decisions) did not previously answer.**
It is written down here rather than left to be re-derived, because "it would only be one file" is
the form the argument usually takes and this is the form it takes when made well.

The argument. Derivation-based tests and cross-language differential tests answer *different*
questions: the first asks whether an object matches the theorem, the second asks **where two
implementations of the intended algorithm first diverge**. The second is diagnostic rather than
evidential — the proposal concedes that passing parity would prove nothing — and it is at its most
useful exactly where this repository is stuck: a flat `√n R_remaining` whose cause is unknown, and
an R implementation whose targeted nuisance arrays, per-round score means and fluctuation
coefficients are inspectable. The bounded form was: freeze a no-missingness binary-treatment
fixture, supply identical initial `Q̄` and `g`, export per round and arm, compare `Qsteps=1`
against `Qsteps=2` against this package's two update orders, and localise the earliest divergence.

**It is refused, and the reason is not the one the standing decision gives.** The standing
decision's reason — agreement is evidence about a transcription — is an argument against parity as
*evidence*, and this proposal does not claim to be evidence. The reason it is refused is narrower
and is about what it would cost against what it could find:

- **The divergence it would localise is one this repository already localises without R.** Item 21
  found a sign error in the mechanism correction by reading the paper's appendices, and a parity
  run could not have caught it, since both sides read it off the same display. Item 20 settled the
  uncentred curve in thirty lines by recomputing a recorded number from the returned state in the
  same process. The instrument that finds a construction difference here is the exact law, where
  every term is a finite sum and *both* sides are known — and `tests/discrete_law*.py` already
  compares against a truth that R would only approximate.
- **The specific divergences it names are already answerable in prose.** That the R source assigns
  its `grnStar` output under the key `grn`, and that it defaults to `Qsteps=2` — a backfitting-style
  outcome update — are facts about that source, and recording them where the code came from is
  what the standing decision explicitly *permits*. Neither needs the code to be run.
- **The current diagnosis has a cheaper instrument that has not been exhausted.** E2R's decision
  run is owed and separates reduction error from estimator construction directly. Reaching for a
  second implementation before the experiment that was built for the question has been run is the
  reverse of the order [WP1](#the-reboot-and-the-three-work-packages) sets.

**What would reopen it**, since a refusal with no such condition is a prejudice: E2R coming back
`unresolved` *and* E3's construction diagnosis failing to localise the disagreement on the exact
laws. At that point the question stops being "does this match the theorem" — which the laws
answer — and becomes "which of two readings of an ambiguous appendix is implemented", which is the
one question a second implementation can settle and a derivation cannot.

## What the sizings got wrong

Sixteen lessons, and they now live in [the investigation
log](drtmle/investigation-log.md#what-the-sizings-got-wrong) with the rest of the record. They are
kept because the only thing a retrospective is for is the next sizing, and they are not on this
page because a plan is not a history. In one line each:

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
