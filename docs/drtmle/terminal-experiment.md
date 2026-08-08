# F5: the terminal experiment

[The roadmap](../roadmap.md#f5-the-terminal-experiment) puts **F5** last: it is the final
statistical experiment in the `DRTMLE` investigation, and the plan ends in a promotion or a stop
rather than in another row. This page is its record.

**Status: the harness is built and the design is frozen; no inferential fit has run.** What is
recorded below is the design, the arm matrix, the decision rule and the two readings that were
taken to *shape* the experiment rather than to answer it. The verdicts, the nomination and the
confirmation are not here yet, and nothing on this page may be read as a result.

## What the experiment asks

F4 landed and returned a null — no construction factor localizes the remaining calibration
shortfall — and C3c had already established that the corrections do most of the work they were
built to do (`74.6%` and `78.7%` of `TMLE`'s coverage deficit recovered, in two independent
batches, paired on the draw). So the open question is no longer *why is `DRTMLE` no better than
`TMLE`* — it is better — but the narrow one:

> Can correctly or feasibly estimated reduced regressions close the remaining calibration gap,
> and does the answer depend on pooled against nested cross-fitting?

[The concordance](theorem-concordance.md) marks the shipped `glm` reductions' consistency
**`unverified`**, which is the premise C3c held fixed and never tested.

## The arms, and why there are five rather than eight

The roadmap's matrix crosses four reduced learners with two cross-fitting constructions. Five of
those eight cells run, and each dropped cell has its own recorded reason. **Two were dropped
before any inferential fit and one was withdrawn after 41 partial draws had been read**, and
that distinction is the most important thing on this page — it is worded differently below
because it *is* different.

| arm | reduced learner | `reduced_crossfit` | role | nominable |
| --- | --- | --- | --- | --- |
| `glm-pooled` | `glm` | pooled | the shipped baseline, and C3c's | no |
| `glm-nested` | `glm` | nested | construction comparator | no |
| `gam-pooled` | `gam` | pooled | feasible candidate | **yes** |
| `gam-nested` | `gam` | nested | feasible candidate | **yes** |
| `ceiling` | reference reductions, `spline(16)` | — | ceiling | no |
| ~~`ceiling-nested`~~ | | nested | *null by construction, dropped before any fit* | — |
| ~~`boost-pooled`~~ | `boost` | pooled | *dropped on the pilot, before any fit* | — |
| ~~`boost-nested`~~ | `boost` | nested | *withdrawn on cost, after partial draws* | — |

**`boost-pooled` was dropped before any inferential fit, on the timing pilot alone.** The
roadmap marks it *diagnostic only, not nominable*:
[A1b's argument](../roadmap.md#a1b--the-cross-fitting-construction) carries the pooled
construction on a one-dimensional bounded-variation ball or a fixed-dimension sieve, a boosted
reduction's pooled design/target-continuity premise is not closed, and so `boost` reaches a
production branch under `nested` or not at all. The pilot measured it at **344 s a fit at
`n = 600`** against the baseline's **5.3 s**, so it was most of a phase-1 budget spent on an arm
no branch of the terminal plan can read.

**`boost-nested` was withdrawn later, and on cost — not on its readings.** After 41 partial
selection draws it stood at **77% of a draw** and roughly **13 h of a 17 h** phase-1 budget. The
scope decision behind the withdrawal is that F5 asks whether `DRTMLE` is *constructed* correctly,
and a boosted reduction is one candidate for that rather than the question itself.

**It is not withdrawn because those draws looked bad, and that distinction is not a formality.**
They did look bad. Withdrawing an arm *because* of what its own decision columns showed is
selecting on the outcome, which is the failure mode a preregistration exists to prevent and
which [stop-ship 17](../roadmap.md#stop-ship) names. So no reading taken on `boost-nested` is
carried forward, reported as a result, or used to support any conclusion — including the
tempting one that a boosted reduction does not help. **F5 makes no claim about boosted
reductions at all.**

**What the withdrawal costs, stated rather than absorbed.** F5's learner screen is now a
fixed-basis smoother against the shipped GLM against the ceiling, and nothing else. The
cross-fitting axis — which exists because F4 measured something on it, `nested ~ cleverly`
moving the point estimate in seven readings of eight — stays identifiable at both `glm` and
`gam`, so the learner and the construction remain separable. `prereg.json` carries all three
drops under `phase1.arms_dropped`, so the deviation from the roadmap's matrix is in the artefact
and not only on this page.

**`ceiling-nested` is a null by construction, and that was measured rather than argued.**
`reduced_crossfit` acts only inside `fit_reduced`,
through `NuisanceEstimates.inner`, on the reduced regressions' *training* rows;
`ReferenceReductionDRTMLE` replaces the produced set in `_nuisances` and replaces `spec.refit`
in `_reduction`, so no array `fit_reduced` produces is ever read. `ceiling_crossfit_reading()`
fits both keywords on one deterministic tier-2 draw and compares every reduced array element by
element, at both ends of the alternation:

| stage | `qr` | `gr1` | `gr2` |
| --- | --- | --- | --- |
| initial | identical | identical | identical |
| converged | identical | identical | identical |

Running that cell would have spent a share of the phase-1 budget measuring a zero, which is the
failure mode F4's own row names — *a declared stress design that cannot be stressed is worse
than no stress design: it spends fits and reads as evidence*. If the equality ever fails, the
arm comes back: `tests/unit/test_drtmle_f5.py` asserts it, in the fast tier rather than behind
the `slow` marker, because a check that decides the shape of the experiment must not be one a
default selection skips.

**LightGBM is required and the fallback is refused.** `library._boost` returns
`HistGradientBoosting` under the same `"boost"` name where LightGBM is absent, and two function
classes under one arm name would make the entropy row of
[the contract](../roadmap.md#the-supported-contract-and-item-25) ambiguous about which was
fitted. `refuse_on_fallback()` is called before any phase, inside `validate_prereg`, and once
per worker; every recorded row also carries the resolved implementation, so a fallback that
somehow got fitted is visible in the artefact and not only in a preflight.

**Every flexible library keeps `mean`.** The shipped baseline *is* the `"glm"` preset, which is
`mean + glm`. A single-candidate `gam` arm would move the function class **and** the ensemble
shape — two factors, which is exactly what F4's matrix exists to prevent. Whether an arm then
collapses onto `mean` is a recorded column (`flex_weight_min`) rather than an inference.

## What the timing pilot measured, and what it may be used for

The pilot ran on the reserved third seed stream — `SeedSequence(20260201).spawn(3)`'s child 2,
which leaves the two cohorts byte-identical to what `spawn(2)` would have produced — and every
row it wrote carries `cohort="sizing"`. **It measures cost and nothing else.**
`contrast_rows()` and `nominate()` raise on a sizing row, so that is structural rather than
remembered: sizing comes from F4's committed `PILOT_PAIRED_SPREAD` and nothing else, which is
what F4's row grants F5 and the whole of it.

8 draws, 56 fits, **45.0 min wall at `--jobs 8`**, zero errors — every arm fits at both sizes in
both cells. Median seconds per fit:

| arm | `q-drift` 600 | `g-drift` 600 | `q-drift` 2,400 | `g-drift` 2,400 |
| --- | --- | --- | --- | --- |
| `ceiling` | 19.0 | 9.3 | 34.1 | 25.6 |
| `glm-nested` | 17.6 | 17.5 | 63.4 | 65.8 |
| `glm-pooled` | 18.9 | 26.2 | 55.1 | 41.4 |
| `gam-nested` | 24.7 | 29.0 | 65.6 | 91.3 |
| `gam-pooled` | 41.8 | 49.5 | 54.2 | 50.9 |
| ~~`boost-pooled`~~ | 959.2 | 948.1 | 1303.2 | 1294.0 |
| `boost-nested` | 946.6 | 931.8 | 1111.7 | 1107.5 |

**These are contended readings and are not the cost column.** Eight workers ran throughout, and
against serial single-draw runs of the same arms the inflation is a fairly uniform ~3x
(`glm-pooled` at `q-drift` 600 reads 5.3 s serially against 18.9 s here; `boost-pooled` 344.5 s
against 959.2 s; `ceiling` 6.4 s against 19.0 s). A runtime taken under eight-way contention is
partly a measurement of the scheduler, so the reported cost column is taken serially by
`--phase cost` and these numbers size the wall clock instead.

The model checks out against the run: with draws ≤ workers every draw runs concurrently and each
draw's arms run serially, so wall clock should equal the longest single draw. The longest is
`q-drift` at 2,400 — 2,687 s summed over its seven arms — which is 44.8 min against the 45.0 min
observed.

**Two things it settled.** The ceiling arm is **7x cheaper** than planned (its reference is a
weighted least-squares projection, not a fit), and `boost` is **23x more expensive** —
`boost-nested` alone is 80–89% of a six-arm draw's entire cost. Dropping `boost-pooled` roughly
halves phase 1; what remains is still dominated by the one boost arm, and the lever if that ever
has to move is LightGBM's `n_estimators`, recorded as a deviation.

At these readings phase 1 is about **19 h** — 9.5 h per cohort, two cohorts — against the 5 h
the plan assumed before anything was measured.

## The frozen rule

A **non-overlapping** partition with a band per column, declared before the first fit. This is
[F4's rule defect](construction-contrasts.md#two-defects-in-the-frozen-rule-found-by-the-run-and-not-repaired-after-it)
repaired rather than reordered, in both of its halves.

<!-- generated by benchmarks.drtmle_f5.format_rule_table(); do not hand-edit -->
```
column            group      band                          scale     better  statistic  gates
----------------  ---------  ----------------------------  --------  ------  ---------  -----
root_n_remaining  theorem    q-drift 0.125, g-drift 0.413  absolute  lower   paired     yes  
score_8           theorem    0.0001                        absolute  lower   paired     no   
score_9           theorem    0.0001                        absolute  lower   paired     no   
score_10          theorem    0.0001                        absolute  lower   paired     no   
score_failures    theorem    0.02                          absolute  lower   paired     yes  
bound_active      theorem    0.01                          absolute  --      cohort     no   
abs_error         estimator  0.1                           relative  lower   paired     no   
root_n_bias       estimator  0.1                           relative  lower   cohort     no   
rmse              estimator  0.1                           relative  lower   cohort     no   
empirical_sd      estimator  0.1                           relative  --      cohort     no   
se_ratio          estimator  0.1                           absolute  lower   cohort     no   
abs_coverage_gap  estimator  0.05                          absolute  lower   cohort     yes  
risk_qr           mechanism  0.1                           relative  lower   paired     no   
risk_gr1          mechanism  0.1                           relative  lower   paired     no   
risk_gr2          mechanism  0.1                           relative  lower   paired     no   
risk_h3           mechanism  0.1                           relative  lower   paired     yes  
risk_h2           mechanism  0.1                           relative  lower   paired     yes  
seconds           cost       1                             relative  --      cohort     no   
```

**The predicate cannot drift from this table**: it is generated from the same `Column` tuple the
verdict reads, and `tests/unit/test_drtmle_f5.py` compares the committed block byte for byte.
One line of the roadmap once named two primary outcomes while the code banded one of them; that
cannot recur here.

### The two defects, and how each is made unreachable

**The clause collision.** F4's `_verdict` tested `moved` before `flat`, so an interval like
`[-1e-4, -1e-5]` — excluding zero, and lying wholly inside its own `±0.125` margin — was
labelled a localization. `verdict()` here tests `equivalent` on a two-sided predicate that a
beyond-verdict contradicts by construction, so at most one can fire. That exact interval is a
regression test.

**The unreachable third verdict.** F4 passed `float("inf")` as the band for every column but
one, so `unresolved` never appeared in five of six columns. There is no per-call band argument
here at all — the band comes off the `Column` — and a missing or non-finite band is a refusal in
`validate_prereg`.

### Three bands could not be anchored absolutely, and that is a finding about the record

C3c committed **no RMSE** for `ate` in any cell, at any size, in either batch — it appears in
the repository only as a column F5 is *required to report*. Empirical SD and mean reported `se`
are committed only for `q-drift`, `n = 2,400`, batch A; root-`n` bias only for `q-drift`, batch
A. Inventing an absolute band for those would be an invented threshold wearing a citation, so
they are declared **relative** — a tenth of the baseline arm's own realized value in that cell
and size — before the first fit.

Two more choices that a reader could otherwise mistake for arithmetic:

- **`root_n_remaining`'s band is per cell.** The column sits at `1.25` in `q-drift` and `4.13`
  in `g-drift`; one absolute band across both would be a band for one of them. The *fraction* is
  F4's single `0.10`, applied to each cell's own committed level, which is what keeps the two
  studies on one scale. `q-drift`'s `0.125` is F4's `NEGLIGIBLE_EFFECT × C3C_REMAINING_QDRIFT`
  carried verbatim. C3c's two batches read `1.252` and `1.174`, so a two-batch anchor would give
  `0.117`; carrying F4's is the choice, on comparability, and the difference changes no verdict
  at this resolution.
- **Phase 1 cannot resolve a coverage move smaller than about `0.10`.** At 24 and 80 draws the
  Monte Carlo error swamps a `0.05` band, so `abs_coverage_gap` will read `unresolved` in most
  phase-1 cells. That is correct and harmless: coverage in phase 1 is a **veto only**, and
  `unresolved` fires no veto. Coverage is settled in phase 2, at 500 replicates per batch.

### Identity failures are exact, not banded

C3c recorded **zero** identity failures across all 6,000 fits. Turning that into a paired
difference with a tolerance would make the one quantity the study has an exact answer for into a
statistical one, so it is not a declared column: any nonzero count on any arm is a veto, and a
stop-immediately condition besides.

### `glm-nested` was wrongly barred from nomination, and that is corrected

The roadmap's phase-1 text excludes exactly one thing by name: *"a ceiling arm may not be
nominated, because it is not a procedure a caller can run"*, with `boost-pooled` separately
fenced by A1b. It says nothing about `glm-nested`. The first freeze nevertheless marked that arm
non-nominable, on a reading of the roadmap table's label *construction comparator* as though a
label were an exclusion. It is not.

`reduced_crossfit="nested"` is a shipped keyword; a caller can run it; F7 could promote it by
moving that default. And barring it cost more than one arm: F5's question has two halves — *can
better reduced regressions close the gap, **and does the answer depend on pooled against nested
cross-fitting?*** — and as first frozen, F5 could *measure* the cross-fitting axis but never
**nominate** on it. A cross-fitting-only improvement could have been observed and then not
recommended.

**The timing is disclosed rather than smoothed over.** The correction was applied after 41
partial selection draws existed and after their `√n R_remaining` readings had been reported —
readings on which `glm-nested` looked favourable. The argument for the correction is
data-independent, and the design intent behind it predates any result; but the sequence is what
it is, and a reader is entitled to weigh it. Two facts bound what the correction could have
done: `nominable` is read in exactly **one** place, `nominate()`, and by no fit path — so no
draw, no estimate and no recorded column moves — and the decision **rule** block of the manifest
is byte-identical, since eligibility is a design field and not a band.

`glm-pooled` stays non-nominable, and that one is structural rather than a choice: it *is* the
shipped default, so F7 promoting it would change nothing and the branch it lands on is the stop.
It is also the arm every contrast is read against.

The manifest now carries `phase1.nominable` and `phase1.not_nominable` with a reason apiece. The
first one did not — `arms` held names only — so a reader could not have checked which of them a
nomination was permitted to select. That was its own gap and is closed with this.

### One clause was corrected before the cohort was read, and the correction is on the statistic

Nomination clause 5 asks that the flexible candidate carry real weight, so that an arm which
collapsed onto `mean` is not nominated as the baseline under another name. As first frozen it
read **`flex_weight_min`** — the minimum over three reduced regressions and five folds, fifteen
values — and asked *that* to clear `0.05`.

That is not the question it meant to ask. A single fold in which an ensemble puts no weight on
the flexible candidate is ordinary, so the minimum sits at zero for almost any arm. Measured on
the first 41 partial draws it read `0.0000` at **every quantile including the maximum** for the
boosted arm, and at the median for both spline arms. **No arm could have passed**, so the study
would have returned *no nomination* — a terminal stop under the branch table — as an artefact of
its own predicate rather than as a statement about any estimator.

That is F4's rule defect rebuilt under a new name, which is precisely what F5's row exists to
prevent, so it is repaired on the instrument rather than absorbed. The clause now reads the
candidate's **mean** weight; `flex_weight_min` is retained beside it as a diagnostic that no
clause reads, so the correction is visible in the artefact rather than taken on trust. **`0.05`
and `0.90` are unchanged** — only the quantity they are applied to moved, which is what
separates this from a threshold moved to clear.

This is the [F3-closeout](../roadmap.md#the-eight-pull-requests) precedent — *the instruments
corrected before anything reads them* — and the correction landed before any verdict was taken.

### A veto that cannot fire is refused

`refuse_dead_gates()` runs before a nomination is taken and **raises** if any gating column has
no finite reading anywhere. This is F4's second defect one layer down: a column whose *values*
are absent produces `nan` intervals and therefore `unresolved` everywhere, which fires no clause
and reads in a table exactly like a veto that fired and found nothing.

## The ceiling arm, and the two errors it has

The frozen rung is `SplineProjection(n_knots=16, degree=3)`, and it is chosen on E2R's **audit**
rather than on its selection cohort. That cohort picked `spline(8)` in ten of twelve
`(cell, size, regression)` rows, but the audit resolved exactly one rung-against-rung comparison
and it went the other way: at `q-drift`, `n = 2,400`, on `q_r`, `spline(16)` measurably beat
`spline(8)`. The other three cells failed only the non-inferiority clause, which is *not shown
equal* rather than *shown worse*. `spline(16)` is also E2's shipped rung and the one
[the manifest](study-manifest.md) already names, so freezing it means this arm inherits E2's
identity rather than inventing one.

**Two error sources, two instruments, and conflating them is how this arm gets over-read:**

- its **numerical** error — the reference's own Monte Carlo — is the across-scramble spread of
  `psi` over independent reference randomisations, read against the smallest decision margin in
  the frozen rule (`0.02`). It is **not** a coarse/fine refinement pair; that statistic is
  [withdrawn](../roadmap.md#what-e1-landed-and-what-e1b-withdrew) and F5 may not inherit a
  retracted statistic as its fidelity gate;
- its **smoothing** bias at tier 2 is bounded by nothing F5 can run. `held_out_risk` estimates
  `C + ‖m − f‖²_w` with `C` common to every candidate, so the reference gate observes a
  *difference* and never an absolute — which is [F6](../roadmap.md#the-eight-pull-requests)'s
  open question, and F6 runs beside F5 rather than before it.

So the arm is reported as a **ceiling estimate** and never as an oracle unless F6's
absolute-adequacy route lands and its anchor covers these regressions. On the exact law it *is*
an oracle, and the exact-law anchor is what separates *the construction is right* from *the
smoothing at tier 2 is adequate*. **A ceiling arm that cannot meet the numerical-error rule is
reported unresolved on that arm** and is never promoted to an oracle for being the best number
available — and two of F5's six branches are ceiling branches, so that outcome narrows what the
confirmation can conclude.

One constraint worth recording because it raises mid-cohort rather than at construction: the
reference block's floor is on **points**, not rows. `spline(16)` has 19 basis columns and
budgets 64 rows each, so it refuses a fit on fewer than 1,216 rows — and `q_r`'s fit is masked to
one arm, while a quadrature block interleaves the two arms one row each. A block of `P` points
therefore offers `2P` rows but only `P` of them to `q_r`.

## F8 clause 3 is recorded not feasible

Phase 2 carries all ten of F8's clauses. **Clause 3 — a reproduction of the published paper's
relevant simulation setting *where feasible*, and a recorded statement if it is not — takes its
escape hatch**, and the three lines of the record that say why are cited rather than summarised:

- [`docs/references.md`](../references.md) marks Benkeser, Carone, van der Laan & Gilbert (2017),
  *Biometrika* 104(4):863–880 — the published paper — **"Not read here"**;
- [the concordance](theorem-concordance.md) marks the same source `in repository: no`, and its
  coverage table lists what *was* read first-hand from the 2016 working paper: Theorem 1, the
  corrected influence function, appendix B's remainder terms, appendix A/B's rate conditions, the
  empirical-process conditions and the recursive algorithm. **No simulation section appears in
  that list**;
- [the investigation log](investigation-log.md) records the reason: only the working paper's text
  — "Theorem 1 and appendices A to C" — is obtainable without a Biometrika subscription.

Reproducing "the published paper's simulation setting" would therefore mean inventing a
data-generating process and attributing it to a paper nobody here has read. That is a fabricated
citation, and no amount of care in the rest of the study repairs one.

**What runs in its place discharges a different committed clause that C3c left unread** — gate 2
clause 4, *the advantage persists in at least one applied stress setting*
([the validation plan](validation-plan.md)). `benchmarks/drtmle_stress.py` is `nonlinear_dgp()`
with the `"fast"` preset on **both** primaries: both nuisances are misspecified for a GLM by
construction and estimated at whatever rate a flexible library achieves, which is the situation a
caller is actually in — as against the drift cells, where the misspecification is engineered at a
prescribed `n^{-α}`.

The declared misspecification is **asserted rather than described**, and on the statistic the
rate conditions are stated in — the excess risk against the law's own function, not the held-out
risk against the observed target:

| nuisance | glm excess | `fast` excess | ratio |
| --- | --- | --- | --- |
| `g` | 0.01838 | 0.00554 | **3.32x** |
| `Q̄(1, W)` | 0.00377 | 0.00058 | **6.49x** |
| `Q̄(0, W)` | 0.00661 | 0.00063 | **10.53x** |

The distinction is not pedantic: measured as a Brier score the mechanism's ratio reads `1.06x`,
diluted by the irreducible Bernoulli variance every candidate shares. A floor placed on that
would have been a floor on mostly noise.

Three things this cell is **not**, each declared before the first fit: its coverage is
descriptive and is **not** a release number; its remainder is **not** read against clause 4's
vanishing trend, since with both primaries inconsistent no theorem predicts one — **item 13
closes on the drift cells and only on them**; and it carries no ceiling arm, because no branch
reads one there.

`weak_overlap_dgp` was the obvious alternative and is **rejected on scope**: its propensities
crowd hard against `0` and `1`, so nearly every fit would exit bound-active and outside
[§7's scope](theorem-concordance.md#7-truncation-is-not-in-the-theorems-algorithm), and a cell
that can support no theorem-backed claim is not the cell to spend a stress budget on. That claim
is measured too — over 20% of its quadrature grid sits outside `[0.025, 0.975]` against under 2%
of the stress cell's.

## What has not run

The nomination, both cohorts, the confirmation and every verdict. `--phase confirm`,
`--phase readout`, `--phase verify` and the ceiling adequacy statistic are unbuilt. The
mechanism component-risk columns are not yet computed, which is why `refuse_dead_gates()` exists
and why a nomination cannot currently be taken.
