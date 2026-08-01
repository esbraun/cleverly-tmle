# Roadmap

What has landed, what is open, and why native acceleration is not worth building.

**One thing is open**: `DRTMLE`, the doubly-robust-inference variant, which is written and
tested and not finished. [What is still open](#what-is-still-open) is the list, grouped into
four pieces of work, each of which is a pull request rather than an errand. Everything else on
this page is a record: [Refusals worth lifting](#refusals-worth-lifting) is the list of
parameters this package had the machinery for and had simply not written down, and it is now
empty; [What the sizings got wrong](#what-the-sizings-got-wrong) is what estimating that work
taught, kept because the next sizing is the only thing it is for.

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
  close** at the sizes it could reach; and **under weak overlap the score check fails on 23 of
  24 fits**, so do not use this estimator where overlap is poor. [What is still
  open](#what-is-still-open) is the rest

## What is still open

**Done means one thing: a demonstration that `DRTMLE`'s interval attains its nominal coverage
where a plain `TMLE`'s does not.** That is [piece C](#c-the-demonstration) below, and nothing
less clears the variant. The other pieces are preconditions for believing such a demonstration;
the [limitations](#limitations-recorded-rather-than-fixed) after them are real, understood, and
would not move a coverage number.

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
opened, because `benchmarks/bench_drtmle.py`, `.github/workflows/drtmle-convergence.yml` and
`estimators/targeting.py` all cite them by number. The pieces are lettered so the two cannot be
confused.

### The work, in four pieces

Each is a pull request. Small items are grouped where the *evidence* is shared — piece B is
three items because one dispatch of the same sweep answers all three — not where the subject
matter merely rhymes.

#### A. Check the curve against something other than itself

**Closes items 1 and 2.** The influence curve `D = D* − D*_Q − D*_g` is read off `drtmle`'s
implementation, not derived. The whole variant is a variance estimate, so a curve transcribed
from software and never checked against its derivation is the one part of this that could be
wrong in a way nothing here would catch. `inference/influence.py::reduced_corrections` says so
in its own docstring, as do the guide and the appendix.

Two halves, and they are one piece of work because they answer the same question:

- **Read Theorem 1 of Benkeser et al. (2017).** The only thing on this page that has to happen
  outside the repository: Biometrika is paywalled and this environment's network policy denies
  the working-paper mirrors. If the theorem and the transcription disagree, the theorem wins
  and `reduced_corrections` is wrong.
- **Cross-check one fit against `drtmle`'s own output.** The cheapest check that would catch
  most of what the first half is about, and the deliverable is small: one fixture fit run in R,
  its `psi` and `se` committed, one test. The package has no cross-language test anywhere, so
  this is not a new gap — but it is a more costly omission here than elsewhere, because the
  whole estimator is a transcription of that package.

**One trap for anyone reading the R source alongside the paper**, and it is why this piece is
not the errand it looks like: `grn1` there is the paper's `gr2` and `grn2` is the paper's
`gr1`. The numerator and denominator roles are swapped between the two, so a formula
transcribed from one and checked against the other is inverted and still plausible.

When it lands, the labels change with it: `reduced_corrections`, the [methodology
section](methodology.md#doubly-robust-inference-what-the-extra-equations-remove) and the guide
all currently say **what `drtmle` computes** rather than what the theorem derives, and that
wording is load-bearing until this piece closes.

#### B. Weak overlap, and the loop's exit under the rule it now uses

**Closes items 11 and 12, and re-measures items 4 and 6.** Three things, and they are one piece
because one dispatch of `benchmarks/bench_drtmle.py` produces the evidence for all of them.

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

**This piece goes first**, because poor overlap is a natural way to make the remainder bite and
so may be exactly where piece C has to look — and it is currently the one regime where the score
equations are not solved at all.

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
  a difference, and it is what rules out a coincidence at a single `n`.

One trap in building it, already met once: `tests/e2e/test_double_robustness.py`'s "correct"
cell is an **oracle** (`OracleOutcomeContinuous`, `OracleTreatment`), which makes the good
nuisance exactly right, `R₂` exactly zero and `TMLE`'s interval already valid. The gap opens only
where the good nuisance is *estimated*.

**What it costs, since that is why it has not been done.** A `DRTMLE` fit is 43s at `n = 1,200`
(measured, [the sweep](#how-the-alternation-exits)) and a study runs both estimators over every
replicate. Two cells by two sizes by 250 replicates is ~2,000 fits, which is ~24 hours serial
and about two on a 12-way `matrix:`. That is a dispatch-only workflow of its own —
`drtmle-convergence.yml` is the template — and the nightly tier must not absorb it.

#### D. Widen the scope to what the sources derive

**Closes the two candidates in item 10.** Everything else in that item is a refusal with a
reason, not a gap. Both candidates are gated on reading rather than on writing, which is why
they sit behind piece A rather than beside it.

- **`reduction="bivariate"`.** van der Laan (2014)'s original single bivariate
  `gr(a | w) = P(A = a | Q̄̂(a, W), ĝ(a|W))` in place of the `gr1`/`gr2` pair, with equation
  (10′) in place of (10). It is derived in the sources and was in scope; it was cut because it
  is a different extra equation on a two-column design rather than a wider loop over the first,
  and nothing was waiting on it.
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
  `ipsi` declares `requires_binary_treatment` and has never needed one.

**The order to work in**, which follows from the above rather than from effort: **B**, because it
may be where the demonstration has to happen and it is currently broken; then **A**, which is
cheap and buys most of the assurance the theorem would; then **C**, which is the point. **D** is
independent of all three and should not queue behind them.

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
term keeping the reported curve's mean off machine zero.

**6. The closing pass's mechanism stage stops on its cap, not on its tolerance.** It settles
around `1e-9` rather than reaching `spec.tol = 1e-10`, on **94 of 96** swept fits. Harmless —
the steps are arithmetic, and item 5 is why it cannot get there — but a cap that always binds is
worth knowing about rather than reading as convergence. The two fits that stopped otherwise are
both `weak-overlap`, which is what piece B is about.

**8. `retarget` is no longer arithmetic on cached arrays.** The reductions are refitted inside
the alternation, so a truncation curve or an MNAR sweep costs about a fit per point rather than a
fraction of one, and a result read back from disk cannot retarget at all — its estimator is gone
and there are no learners to refit with. `ReductionFluctuation` is not serialised either, so a
reloaded fit keeps its estimates and loses the record of what solved them. This is a cost of
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

Kept as two lines each rather than deleted, because both are cited by number from
`benchmarks/bench_drtmle.py`, `.github/workflows/drtmle-convergence.yml` and
`estimators/targeting.py`.

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

**12. That change is now pinned by a test.** It was not, for a while, and the gap is the one
`CLAUDE.md` names: the whole 61-test `drtmle` suite passed identically before and after, because
every assertion in it is about the *reported* fit and the closing pass makes that fit the same
either way. `TestAnEquationStopsOnEitherRuler` in `tests/unit/test_drtmle_fit.py` unit-tests
`_solved` directly, and the absolute branch was deleted and the suite watched to fail — two of
the four assertions go red — before the test was kept. Asserting `exit_reason == "tolerance"` on
a fitted result was rejected: which exit fires is a property of the draw. The item's two other
loose ends are open and are in [piece B](#b-weak-overlap-and-the-loops-exit-under-the-rule-it-now-uses).

### How the alternation exits

96 fits: four processes by two sizes by twelve seeds, `glm` on both nuisances, `n_folds=5`,
`learner_folds=3`, both the data seed and the fold seed varying. Dispatched as
`.github/workflows/drtmle-convergence.yml` from `benchmarks/bench_drtmle.py`, 2,588s of runner at
42.6s per fit, and **no fit raised**. The rows are kept here for the reason `bench_tmle.py` keeps
its own: a comparison nobody can rerun becomes folklore.

**These numbers measure the exit criterion item 7 replaced**, not the current one. That is
deliberate and is the order the item required — the failure had to be characterised before the
threshold moved — and re-measuring under the current rule is part of [piece
B](#b-weak-overlap-and-the-loops-exit-under-the-rule-it-now-uses).

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

Five lessons, distilled from the per-item retrospectives that used to run to several hundred lines
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
