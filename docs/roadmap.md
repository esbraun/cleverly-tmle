# Roadmap

What has landed, what is open, and why native acceleration is not worth building.

**One thing is open**: `DRTMLE`, the doubly-robust-inference variant, which is written and
tested and not finished. [What is still open](#what-is-still-open) is the list, grouped into
pieces of work, each of which is a pull request rather than an errand.

That grouping and its order are a revision, three times over. An [external
review](drtmle-review.md) of this page and the code behind it read the plan against Theorem 1 of
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
preceded and depended on pieces A1 and A2 — so B1 splits into **B1a**, an identity and safety
patch valid under every eventual convention, and **B1b**, the targeting decision that has to wait
for the theorem.

**Then the working paper itself arrived**, at `docs/viewcontent.cgi.pdf`, and reading it rather
than a transcription of it closed both of the third review's items — the **sign in favour of the
implementation**, on that paper's own appendices, which contradict the display the charge was
filed from; and the **update order**, because the paper states its own exit as the three score
equations and so prescribes a fixed point rather than a route. That is
[lesson 10](drtmle-investigation-log.md#what-the-sizings-got-wrong), and it is why this page no
longer opens with a stop-ship.

**This page is now one of four**, because it had become a status page, a methodology review, a
forensic report, an implementation specification, a test plan, a simulation protocol and a project
history at once, and that density made the dependency plan harder to execute than the plan itself
was:

| document | what is in it |
| --- | --- |
| **this page** | status, the pieces, what blocks what, the release gates, the definitions of done |
| [theorem concordance](drtmle-theorem-concordance.md) | the theorem objects, the assumptions and their grading, the appendix-B terms, the paper/Python/R mapping, the cross-fitting status |
| [validation plan](drtmle-validation-plan.md) | the fixtures, the candidate targeting algorithms, the benchmark columns, the coverage design, the frozen decision rules, the mutations |
| [investigation log](drtmle-investigation-log.md) | item 20's discovery, the clipped-row measurements, the hypothesis that was dropped, the convergence sweep, the runner history, the lessons |

Everything else here is a record: [Refusals worth lifting](#refusals-worth-lifting) is the list of
parameters this package had the machinery for and had simply not written down, and it is now
empty.

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
  finished, and calling it landed would be claiming the part that is missing. **One defect is
  live and it is a stop-ship; the other is closed and is worth keeping visible:**

  - ~~the mechanism correction's sign disagrees with Theorem 1~~ (item 21) — **closed, and the
    curve agrees with the theorem.** The working paper is now in the repository
    (`docs/viewcontent.cgi.pdf`) and read. Its §3.1 display does define `D_A = −(Q_r/g)(A − g)`
    while Theorem 1 subtracts `D_A`, which read together flips the mechanism correction — but the
    paper's **own appendices** derive each block in a form satisfiable only with the *positive*
    correction, and Theorem 1's variance formula then reads exactly as this package computes it.
    So the display is wrong on its face, in a document that also prints `D_Y` twice with two
    signs. The argument, the two further slips in the same paper, and what is left are in [the
    concordance's §4](drtmle-theorem-concordance.md#4-the-sign-discrepancy-item-21--resolved);
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
    log](drtmle-investigation-log.md#item-20-from-discovery-to-cause) carries the measurements.

  So **until [B1b](#b1b--the-theorem-conforming-targeting-decision) chooses a convention, a
  `DRTMLE` standard error should be read as provisional wherever `res.score_verdict` says so.**
  The live defect is now caught *by name*: [B1a](#b1a--the-identity-and-safety-patch) has landed,
  so `res.validation.correction_check()` recomputes each arm's `Pn[w D*_g]` and `Pn[w D*_Q]` from
  the exact returned state, reports the residual against the score the loop recorded and the
  `B_clip` that explains it, and `score_check` marks such a fit invalid in words that say
  *defect* rather than *did not converge*. Before it, the only witness was the influence-curve
  rows being uncentred, which was how it was found at all. Item 21 was caught by nothing here and
  could not have been — it took the source, and the source only settled it because its
  appendices could be checked against arithmetic this repository already had.

  Beyond those: **the influence curve is transcribed from R's `drtmle`, not derived** — the
  working paper closes most of that gap and the published article closes the rest; nothing here
  has been compared against that package's numbers; and a coverage study found **no gap for the
  variant to close** at the sizes it could reach. [What is still open](#what-is-still-open) is the
  rest. Three of its items are there because the [first review](drtmle-review.md) put them there:
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
*believe* a demonstration that meets it. Coverage is one link of five, and each link can hold
while another fails:

1. **Theorem fidelity** — the equations solved and the curve reported are the ones the derivation
   gives, under conditions the fit actually meets. Items 1, 13, 15, 21 and 22. The third review
   reported this link **broken** at the sign; reading the source closed items 21 and 22 in the
   implementation's favour, so what is left of the link is items 1, 13 and 15 — the labelling, the
   empirical remainder rate, and the cross-fitting construction.
2. **Reference fidelity** — the algorithm agrees with `drtmle` component by component, not
   merely at `psi` and `se`, where several differences cancel. Item 2. Note what it cannot do:
   both packages descend from one source, so parity is evidence about the transcription and is
   blind to exactly the class of error item 21 is in.
3. **Independent correctness** — a check that could not agree with `drtmle` by copying it.
   Half of it exists: `tests/unit/test_remainder_drtmle.py` is the exact-law arithmetic. The
   other half is item 13's empirical rate.
4. **Numerical validity** — every required score solved to a statistically negligible order,
   *at the arrays the reported curve is built from*, and a fit that fails to say so somewhere a
   reader cannot miss. Items 11, 12, 16 and 20. The qualification is item 20's: a fit can solve
   all three equations to `1e-11` by its own record and still report a curve whose mean is
   `2e-04`, which is a link-4 failure that announces itself nowhere in the loop's own
   diagnostics. Items 11 and 20 turned out to be **the same failure** — the loud version under
   weak overlap and the quiet one on a quarter of ordinary splits — and the sentence in italics
   above is where the trap is, because *the arrays are the same arrays*. What differs is the
   truncation applied to one of them on the way into two different expressions. "Built from the
   same state" is necessary and it is not sufficient; the checkable form of this link is an
   **identity between each recorded score and a recomputation of the term the curve carries**,
   which is [piece B1a](#b1a--the-identity-and-safety-patch) and **has landed**. The link is
   still broken — the identity fails wherever the bound binds, which is what B1b decides — but it
   is now measured per arm on the face of every fit rather than inferred from an uncentred curve.
5. **Inferential usefulness** — coverage in a regime where the plain interval fails. Item 3.

The first review's summary of this is exactly right and worth keeping in its words: none of the
five implies the others. A `psi` that matches R proves nothing about the variance, which is the
only thing this variant produces; a curve that matches the theorem proves nothing about whether
the alternation solved its equations on a given draw; and the first four together prove nothing
about whether the interval is ever *better* than the one `TMLE` already reports. The third review
adds the corollary that cost the most to learn: **links 2 and 3 are blind in the same place**, so
two checks that cannot fail against the same class of error are one check, however different their
machinery ([lesson 9](drtmle-investigation-log.md#what-the-sizings-got-wrong)).

The [limitations](#limitations-recorded-rather-than-fixed) after the pieces are outside the chain
entirely: real, understood, and unable to move a coverage number. Anything that *can* move one
belongs in a piece, which is where item 20 went after being filed there by mistake — see
[lesson 7](drtmle-investigation-log.md#what-the-sizings-got-wrong).

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

### The work, in four pieces and seven pull requests

A and B are each split into halves, so the four pieces are seven pull requests. Small items are
grouped where the *evidence* is shared — piece B2 is four items because one dispatch of the same
sweep answers all of them — not where the subject matter merely rhymes.

| PR | what it lands | new artefacts |
| --- | --- | --- |
| **B1a** — *landed* | the score/correction identities, the clipping diagnostic, and invalidation when either fails | `cleverly/validation/drtmle.py`; `res.validation.correction_check()`; `identity` and `correction` rows on `score_check`; tests in `tests/unit/test_drtmle_fit.py` and `test_influence_drtmle.py` |
| **A1** | items 1, 15, 21 and 22: the theorem read, mapped, graded, and the sign adjudicated | closes out [`docs/drtmle-theorem-concordance.md`](drtmle-theorem-concordance.md) |
| **A2** | item 2: `drtmle` parity, component by component | `tools/r_reference/export_drtmle_fixture.R`, `tests/reference/drtmle/*.json`, `tests/unit/test_drtmle_reference_parity.py`, `docs/drtmle-r-reference.md` |
| **B1b** | items 11 and 20: the targeting convention, chosen on theorem fidelity rather than parity | the chosen submodel or solver at the `DRTMLE` call sites; the variant comparison table |
| **B2** | items 12 and 19, re-measures 4 and 6, decides the overlap policy | columns on `benchmarks/bench_drtmle.py`, a dispatch of `drtmle-convergence.yml` |
| **C** | items 3 and 13: the demonstration | `benchmarks/drtmle_coverage.py`, `.github/workflows/drtmle-coverage.yml`, `docs/drtmle-coverage-study.md`, per-replicate results |
| **D** | the two candidates in item 10 | its own reduced object, submodel and fixtures |

**The dependency order is the plan, and it is not the reading order below.**

```text
B1a  identity + safety patch ─────────────────────────────────┐
                                                              │
A1  theorem concordance ──┐                                   ├─> B2  sweep ──> C  demonstration
                          ├─> B1b  targeting convention ──────┘
A2  R component parity ───┘

D  independent of all of it, and gated on A1 alone
```

**B1a first**, because every number B2 and C produce is read *through* the reported curve, and
until it lands a share of every cell's fits report a curve the fit did not solve for. It is also
the cheapest thing on this page, and — the point of splitting it out — **it is valid under every
convention B1b might eventually choose**, so it does not wait on the theorem.

**A1 and A2 in parallel with it**, A1 outranking everything because if the theorem and the
transcription disagree, work already landed is work to redo — and on the sign they already do.
**B1b after A1 and A2**, because the convention is a derivation and not a taste. **B2 and then C**
on the corrected implementation. **D** independent of all of it.

The old plan had B1 both first and downstream of A2, which is not a dependency graph. Splitting it
is the third review's most useful structural correction.

#### A. Check the curve against something other than itself

**Closes items 1, 2, 15, 21 and 22, and opens item 13.** The influence curve
`D = D* − D*_Q − D*_g` is read off `drtmle`'s implementation, not derived. The whole variant is a
variance estimate, so a curve transcribed from software and never checked against its derivation
is the one part of this that could be wrong in a way nothing here would catch — and
[it is](drtmle-theorem-concordance.md#4-the-sign-discrepancy-item-21--resolved).
`inference/influence.py::reduced_corrections` says so in its own docstring, as do the guide and
the appendix.

Two halves. They answer the same question and only one of them is blocked; A2 needs neither the
paper nor R's agreement to be worth doing.

**The document-access problem is gone and was never a paywall.** `docs/pdf.pdf` is the 2023
software article and `docs/viewcontent.cgi.pdf` is the **2016 Berkeley working paper**, supplied
by hand and now read first-hand rather than through a transcription — which is what closed item
21, and which changed the answer: the transcription was faithful and the display it transcribed is
contradicted by the same paper's appendices. What remains unobtained is the **published 2017**
article, an NIH author manuscript in PubMed Central as **PMC5793673**, and van der Laan (2014)
Theorem 3, which only [piece D](#d-widen-the-scope-to-what-the-sources-derive) needs. Neither now
gates anything: item 21 was settled on internal consistency plus exact-law arithmetic, and neither
depends on the edition. One runner's network measurements are in [the investigation
log](drtmle-investigation-log.md#what-one-runner-could-and-could-not-reach) rather than here,
because they are a property of an execution environment on a date and this page carried an
obstacle it had inherited for two revisions.

##### A1 — the theoretical audit

*Closes items 1, 15, 21 and 22.* **Items 21 and 22 are closed**; 1 and 15 are what is left of
this piece, and neither needs a document that is not in hand.

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
  package's arrays pinned against the theorem's terms. R's `eval_Dstar_g` is the one column of
  that comparison still missing, and it is [A2](#a2--reference-and-independent-validation)'s.
- **Close out the concordance.** The permanent table mapping each object of the theorem to its
  Python name and its R name, and stating for each: the conditioning variable, the sign, the
  denominator and its truncation, whether the value is initial or starred, whether it is
  arm-specific, and **which score or influence term consumes it**. It is seeded and its `TODO`
  rows are what is left. It is the artefact that makes the next reader's audit cheap rather than a
  re-derivation.
- **Finish the assumption matrix** — one row per condition, with columns `condition | source |
  required for | what the implementation does | evidence | status`, and a status drawn from
  *met*, *met under a stated restriction*, *unverified*, *violated*, *not covered by the
  source*. "Unverified" is a permitted answer and is the point of the column; a matrix with no
  unverified rows on first pass has been filled in from the code rather than from the paper. Its
  present state already has three rows reading *not covered by the source* — hard truncation,
  `K` arms, and composition with `CTMLE` — and one reading *violated*.
- **Settle the update order (item 22).** The theorem states a six-step recursion in a particular
  order; the Python iteration is not a transcription of it. That is not automatically wrong —
  Theorem 1 asks for the final equations and remainders, not a unique order — but it is unchecked,
  and "the fixed points coincide" is a claim with no instrument. Compare the paper's order, R's,
  Python's, the fixed point each reaches and the final three theorem-defined scores at each. **Do
  not compare fluctuation coefficients across algorithms** unless the submodels and the order are
  identical.

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
item 13 is opened here, where the reason for it is, and closed there. **It is now known to be
insufficient on its own**: the working paper's appendix B splits the remainder into `R_{Q,n}` and
`R_{g,n}`, and a total trending to zero can conceal cancellation between them, so the study
reports the branches separately where the DGP permits. The exact terms are in [the concordance's
§5](drtmle-theorem-concordance.md#5-the-remaining-remainder-terms).

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
per-outer-fold reductions, and the current pooled construction with a proof. Agreement with R
settles nothing here — that package predates the construction — so this is A1's work and not
A2's. Keep both tracks: if the answer is "pooled is fine, and here is why", the docstring gains a
paragraph and the item closes; if it is not, the expensive nested version is the reference
implementation to measure the cheap one against, and it need not become the default to be useful.
[The concordance's §8](drtmle-theorem-concordance.md#8-cross-fitting-is-not-covered-item-15) holds
the status.

##### A2 — reference and independent validation

*Closes item 2.* The current plan's "one fixture
fit, its `psi` and `se` committed, one test" is too coarse to catch what it is for: several
differences cancel at `psi`, and `psi` is precisely the quantity all three empirical means being
zero makes *insensitive* to the corrections. Compare **components** — the four fixtures, the
per-component tolerances, the trap list and what A2 must answer beyond parity are in [the
validation plan](drtmle-validation-plan.md#3-the-reference-fixtures-piece-a2), and the six traps
for reading the R source alongside the paper are in [the concordance's
§9](drtmle-theorem-concordance.md#9-six-traps-for-reading-the-r-source-alongside-the-paper).

Include a **deliberately misspecified** fixture, because at the truth `Q_r` and `g_{r,2}` vanish
row by row and a broken implementation agrees with plain `TMLE` and with R alike — the degeneracy
[lesson 2](drtmle-investigation-log.md#what-the-sizings-got-wrong) is about, arriving here for the
third time. And the second half of A2 is the independent check, which is not optional because a
cross-language comparison can reproduce a shared bug: both packages descend from the same source.
`tests/unit/test_remainder_drtmle.py` is already that check for the *guards*; what it does not do
is pin the reported curve's own decomposition against a perturbation of the law the way the
`test_influence_gateaux*` modules do elsewhere — and those modules cannot be reused here,
derivably, because everything the variant adds vanishes on an exact law.

**Where R and the theorem disagree, keep both results in the fixture and mark the discrepancy
explicitly.** The theorem wins for statistical correctness. A2 could not have adjudicated item 21
and a fixture that quietly recorded R's sign as correct would have made it permanent — that
remains the rule, and the sign it would have recorded turned out to be the right one, which is
luck rather than method.

The labels have moved with item 21: `reduced_corrections`, the [methodology
section](methodology.md#doubly-robust-inference-what-the-extra-equations-remove) and the guide
used to say **what `drtmle` computes** rather than what the theorem derives, and they now say the
two agree and what the agreement took. A2 landing on its own changes
none of it, and the three claims stay separate: that Python implements the same algorithm, that
the algorithm satisfies the theorem, and that it helps in finite samples.

#### B. The loop's exit, and whether what it leaves is what gets reported

**Closes items 11, 12, 19 and 20, and re-measures items 4 and 6.** Six things, and they were one
piece because one dispatch of `benchmarks/bench_drtmle.py` produces the evidence for all of them.
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
log](drtmle-investigation-log.md#item-20-from-discovery-to-cause). Three consequences for this
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
plan](drtmle-validation-plan.md#1-the-invariants-piece-b1a), and the last of them is the
degeneracy [lesson 2](drtmle-investigation-log.md#what-the-sizings-got-wrong) names.

`tests/unit/test_drtmle_fit.py::TestTheReportedCurveIsNotAlwaysCentred` must be **rewritten rather
than deleted**: it currently pins the defect's numbers, and afterwards its fixture is the
regression test that the bounds still bind on that draw.

The immediate next coding task on this whole variant is this and not "make the weak-overlap
fixture pass":

> Make every final score and every correction component a provably identical evaluation of the
> exact returned state, expose the clipping discrepancy explicitly, and invalidate inference
> whenever the final reported score is not negligible.

That patch is valid under every eventual theoretical convention. It gives A1 and A2 clean evidence
to decide the final targeting algorithm with, and it prevents B2 or C from producing apparently
authoritative results through an internally inconsistent curve.

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
  [lesson 8](drtmle-investigation-log.md#what-the-sizings-got-wrong)'s pattern in a second place.
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

*Closes items 11 and 20.*

The current defect can be removed under **more than one** targeting design, and an earlier
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
The four candidates, their costs, and the five-criterion decision hierarchy that ranks theorem
fidelity and exact final-score validity **above** R parity are in [the validation
plan](drtmle-validation-plan.md#2-the-targeting-candidates-piece-b1b).

Two things to carry into the implementation. `solve_mechanism` is shared with `ipsi`, which is a
regression surface: **the change belongs at the `DRTMLE` call sites, not in that function.** And
if a bounded convention is adopted, [limitation 6](#limitations-recorded-rather-than-fixed) gets
*worse* rather than better, since a truncated residual is not the canonical logistic score.

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
  as something to investigate rather than as an automatic rejection.

##### B2 — the sweep, on the corrected implementation

*Closes items 12 and 19, re-measures items 4
and 6, and decides the weak-overlap product policy.* One dispatch of `benchmarks/bench_drtmle.py`
produces the evidence for all of it, and it must run **after** B1a at the least: every conclusion
it could draw today is read through a curve that a share of fits have wrong.

**The diagnosis stays widened even though the cause is found.** `1/g` in equation (8) is one of
*five* places weak overlap enters, and B1 accounts for the score failure without saying the other
four are harmless. The five places, the columns the sweep must add — the clipped-row share first
among them — the oracle-reduction run that separates a noisy reduction from a wrong equation, and
the truncation-curve caveat are in [the validation
plan](drtmle-validation-plan.md#4-the-sweep-piece-b2).

*The sweep measures the criterion that was replaced.* [The
table](drtmle-investigation-log.md#how-the-alternation-exits) is the evidence item 7's change was
argued from, which is the right way round — the failure had to be characterised before the
threshold moved — but it means the exit distribution under the current rule is uncharacterised. A
rerun is one dispatch and about 45 minutes, and would say whether `tolerance` is now the norm at
scale or only on the six fits looked at. It re-measures items 4 and 6 for free.

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

**The product decision belongs to this piece**, and B1 changes what it is likely to be. If the
sweep still finds no stable region, `DRTMLE` should refuse or invalidate under weak overlap on a
**predeclared** diagnostic rather than warn — a warning is easy to miss, and this is a method
whose only purpose is inference. The reporting half of that is
[item 16](#closed-since-this-list-opened), which has landed; what this piece adds is the threshold
and the name of the state, decided from evidence. But the evidence that motivated the refusal was
23 of 24 failed score checks, and on present measurement those are the convention mismatch rather
than the estimator breaking down — so **do not predeclare the refusal before B2 re-measures**.
What survives regardless is the ordinary positivity warning, which fires on these fits already
(29% of units outside the bounds on the seed-0 draw).

**19. The alternation's convergence argument proves less than it is read as proving.**
`solve_with_reduction`'s docstring argues that equation (9) is a weighted logistic MLE of `A | W`
and equations (8) and (10) are the outcome quasi-likelihood — separate factors of the likelihood
of `(A, Y) | W` — so each step maximises its own factor with the others held fixed and "the joint
value never decreases". The first review reads the mid-loop refit of the reductions as breaking
that, and it does not: the reductions enter as the *directions* of the submodels, not as values of
the objective, so refitting them changes the next step's direction and leaves the current joint
value where it is, and monotonicity survives. What does not survive is what the argument is used
for. A bounded monotone sequence converges *in value*; that is why the loop terminates, and it is
not why the iterates approach a common zero of three score equations — under a direction that
changes each round, the fixed point of the ascent need not be a stationary point of anything. The
sweep already shows the gap in numbers: **86 of 96 fits stalled** at a point the objective would
not climb from, against 2 that reached the tolerance. So the wording is the fix — state it as an
estimating-equation iteration with empirical convergence diagnostics, keep the monotonicity claim
for what it does buy (termination, and the reason not to restart from `Q̄⁰`), and drop the
implication that stalling is a numerical disappointment rather than the expected exit.

**20. The reported curve is not centred wherever the mechanism truncation binds, and the
fluctuation rows say it is.** Found by checking item 18 and not by looking for it. Over 24 draws —
twelve `repeats=2` fits on `nonlinear_dgp` at `n=600` with `glm` on both nuisances, `n_folds=5`,
`learner_folds=3` — **six** leave `Pn[D*_Q + D*_g]` above `1e-8`, at magnitudes from `2e-05` to
`7e-04` on the scaled outcome, every one exiting on `"tolerance"` with no failure recorded and no
ill-conditioned round. On one such draw equation (9)'s **recorded** score is `3.7e-11` while the
mean of the `D*_g` the curve actually subtracts is `-2.3e-04`. The cause is above; the record is
in [the investigation log](drtmle-investigation-log.md#item-20-from-discovery-to-cause).

This is not a `repeats=` defect and refusing that keyword would misdiagnose it: a draw of a
repeated fit is an ordinary fit, and the affected draws include first draws. It is also not a
`nonlinear_dgp` defect — that process is where it was seen because that is the module's fixture,
and the quarter-of-splits rate is the rate at which an ordinary `auto` bound binds on 600 rows.

**Which is why the finding cost one fit and not a cross-language fixture**, and that is worth
recording against the instinct this page had. Both earlier revisions put item 20 in A2, reasoning
that a divergence between two arrays that should be equal is what a component-by-component
comparison locates. The reasoning was sound and the premise was wrong: there was no divergence
between two arrays. What located it was **recomputing the recorded score from the returned state
in the same process** — thirty lines, one fit, no R — and then finding the recomputation *agreed*
with the record, which is what pointed at the expression rather than at the state. The general
lesson is [lesson 8](drtmle-investigation-log.md#what-the-sizings-got-wrong).

`score_check` **does** catch it, on the *influence-curve* rows, which are computed from the curve
rather than from what the solver recorded — so a fit in this state now says so on its own report
rather than printing an interval like any other. That is item 16 arriving on the first case nobody
constructed for it, and it is the only reason this was seen. Since
[B1a](#b1a--the-identity-and-safety-patch) it is caught *as itself* as well: per arm, per
equation, against the score the loop stored, with `B_clip` reproducing the discrepancy to floating
point and the verdict saying **defect** rather than **did not converge**. The item is still open —
being measured is not being fixed, and which convention replaces the current one is B1b's.

Item **23**, found by the same instrument on the same run, was in this section and is
[closed](#closed-since-this-list-opened).

#### C. The demonstration

**Closes item 3, and item 3 is the definition of done.** A coverage pilot over the off-diagonal of
the misspecification grid put `TMLE` and `DRTMLE` at 0.958 apiece in one cell and 1.000 in the
other — no gap to close. The diagnosis is understood: a correctly specified *parametric* nuisance
converges at `n^(−1/2)`, so `R₂` is `O(n^(−1))` and the product condition never binds. There was
nothing for the variant to fix. `tests/e2e/test_coverage_slow.py`'s `TestDoublyRobustInference`
guards what it can — that the point estimate is still doubly robust, that the interval does not
*cost* coverage, that the standard error matches the spread of the estimates — and says in its own
docstring that it is not a demonstration.

The remainder is `R₂ = ‖ĝ − g₀‖ · ‖Q̄̂ − Q̄₀‖` and a `TMLE` interval needs `√n · R₂ → 0`, so the
regime wanted is one where that product does *not* vanish fast enough while one nuisance is still
consistent. The full design — the two tiers, the drift-coefficient calculation, the regime-entry
columns, the `P₀D̂` evaluation convention, the reporting schedule, the sizes and replications, and
the frozen decision rules — is in [the validation
plan](drtmle-validation-plan.md#5-the-controlled-study-piece-c). What belongs on this page is why
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
  0.95 in **both** off-diagonal cells, and the conclusion reproducing in the second seed batch.
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

**What it costs, since that is why it has not been done.** A `DRTMLE` fit is 43s at `n = 1,200`
(measured, [the sweep](drtmle-investigation-log.md#how-the-alternation-exits)) and a study runs
both estimators over every replicate. Two cells by two sizes by 250 replicates is ~2,000 fits,
which is ~24 hours serial and about two on a 12-way `matrix:`. A third size and the nuisance-rate
columns roughly double it. That is a dispatch-only workflow of its own —
`drtmle-convergence.yml` is the template — and the nightly tier must not absorb it.

#### D. Widen the scope to what the sources derive

**Closes the two candidates in item 10.** Everything else in that item is a refusal with a reason,
not a gap. Both candidates are gated on reading rather than on writing, which is why they sit
behind **A1** — the reading — rather than beside it.

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
  §10](drtmle-theorem-concordance.md#10-the-bivariate-construction) records the status. It gets
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
effort. Piece **0** was first and has landed, and so now has **B1a**; what is left is:

1. ~~**B1a**~~ — landed. It was first because it is the only piece that changes a number every
   other piece reads, because it is valid under every convention A1 and A2 might select, and
   because it is a patch plus its tests rather than a study. [What it
   shipped](#what-b1a-landed).
2. **A1**, whose items 21 and 22 are now closed — the sign in the implementation's favour, the
   update order because the paper prescribes a fixed point rather than a route. What is left of
   it is item 1's labelling, item 15's cross-fitting construction and the assumption matrix, and
   none of it needs a document that is not in hand. **A2 in parallel**, since it needs neither
   the paper nor a decision.
3. **B1b**, once A1 has said which mechanism the theorem's `D_g` is evaluated at — [the
   concordance's §7](drtmle-theorem-concordance.md#7-truncation-is-not-in-the-theorems-algorithm)
   is where that stands — and A2 has said what R does numerically.
4. **B2**, on the corrected implementation, because poor overlap may be where the demonstration
   has to happen and because the exit distribution under the current rule is uncharacterised.
5. **C**, which is the point.

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

Any one of these blocks calling `DRTMLE` finished, and they are the five links restated as things
a reader could check rather than as claims. The first is new and is the one currently true:

1. a correction term disagrees with Theorem 1 or the appendix — this was item 21 and it is
   **closed**: the appendices force the positive reading, which is the one implemented, and
   `tests/unit/test_theorem_drtmle.py` is what a reader checks that against rather than the
   §3.1 display;
2. the algorithm's fixed point is not shown to satisfy the theorem's three score conditions —
   item 22, whose *theoretical* half is closed (the paper's step 7 states its own exit as those
   three empirical means, so the order is not prescriptive) and whose numerical half is A2's:
   whether the paper's order and this one reach the same fixed point on real data;
3. R and Python disagree on a component with no written reason;
4. a stored score and the term the curve carries are not the same functional of the same state —
   this is item 20, it is the one that was true and unnoticed, and since
   [B1a](#b1a--the-identity-and-safety-patch) it is true and **reported**: such a fit fails its
   own score check with a verdict naming the arm and the equation. Caught is not fixed, and this
   stays here until B1b closes it;
5. a required score is not negligible under the predeclared validity rule;
6. `√n · R_remaining` does not trend to zero in either off-diagonal cell, or does so only because
   the two appendix branches cancel;
7. coverage fails in either cell under the controlled study;
8. the invalid-fit rate exceeds its predeclared threshold in the well-overlapped cells;
9. the conclusion depends materially on excluding failed fits after the fact;
10. it does not reproduce in the second seed batch;
11. any document calls the corrected curve efficient under misspecification (item 14's ground,
    which piece 0 cleared and which prose can re-lose);
12. any document calls the curve theorem-derived on the strength of a *display* rather than of the
    appendices — item 21 closed on the second, and the first says the opposite;
13. an unsupported estimand or treatment structure is accepted without a derivation;
14. a weak-overlap interval is reported as valid where the scores fail.

Note what is **not** on this list any more: a coverage gap over `TMLE` of at least 0.05. That is a
product judgment about whether the variant earns its cost, it has no theorem behind it, and it now
lives in [gate 2](drtmle-validation-plan.md#the-decision-rules-frozen-before-the-dispatch).

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

Over [96 fits](drtmle-investigation-log.md#how-the-alternation-exits) the conditioning is **worst
where the reasoning predicted**, which is the part that had never been tested: `gr2` vanishes where
the mechanism is *right*, so the easy process should be the ill-conditioned one, and it is.
`linear` reports an ill-conditioned solve on **5 of 12** fits at `n = 600` and **9 of 12** at
`n = 1,200`, against **0 of 12** for `nonlinear` at `n = 600`. Running out of rounds is a minority
— 8 of 96 — but converging is rarer still: 2 of 96 reached the tolerance and 86 stalled.

**5. Equation (9) is never solved exactly.** Its covariate `Qr/g*` reads the very mechanism it
tilts, so one solve zeroes the score at the pre-tilt covariate and leaves a residual at the
post-tilt one. The closing pass iterates it — to `4e-12` on the exact law and about `1e-9` on a
fitted one — and does not remove it. Equations (8) and (10) *are* exact, so this is the only term
keeping the reported curve's mean off machine zero. **This limitation is bounded, and it is not
item 20.** The two looked like one story — "four to five orders worse on a quarter of draws" — and
are not: the `1e-9` here is the equation the loop poses, measured at the arrays the loop leaves,
and it stays `1e-9` on the uncentred draws too. What is `2e-04` on those draws is a *different*
expression of the same arrays, which is item 20 and closes in B1.

**6. The closing pass's mechanism stage stops on its cap, not on its tolerance.** It settles
around `1e-9` rather than reaching `spec.tol = 1e-10`, on **94 of 96** swept fits. Harmless *at
that size* — the steps are arithmetic, and item 5 is why it cannot get there — but a cap that
always binds is worth knowing about rather than reading as convergence. The qualification this
entry used to carry — that a cap always binding and `D*_g` being wrong by `2e-04` are "close
enough together to be one story" — was a guess and has been checked: they are **not** one story.
The stage does bind on the mechanism, and the uncentred draws are the ones where the tilted `g*`
leaves the bounds, but the cap binds on 94 of 96 fits while the curve is uncentred on a quarter of
them, so the cap cannot be what selects them. The two fits that stopped otherwise are both
`weak-overlap`. If B1b adopts a bounded-residual convention this entry gets *worse* rather than
better — a truncated residual is not the canonical logistic score — and that cost is priced there.

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
remainder and two over-correct — is stated in exactly those terms. It was independent of A1, A2
and B1b and did not queue behind them.

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
fitted result was rejected: which exit fires is a property of the draw. The item's two other loose
ends are open and are in [piece B2](#b2--the-sweep-on-the-corrected-implementation).

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

Ten lessons, and they now live in [the investigation
log](drtmle-investigation-log.md#what-the-sizings-got-wrong) with the rest of the record. They are
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
   a reference implementation is blind in exactly the same place;
10. a display is not a derivation — when a source and an implementation disagree, check whether the
    source disagrees with *itself* before changing the code, because item 21 did.

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
