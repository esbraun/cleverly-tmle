r"""Is item 13's flat remainder a *learner* failure at all?  The gates, before the number.

``docs/roadmap.md``'s **E2**, repaired as its **E2R**.  C3c read
:math:`\sqrt n R_{\text{remaining}}` as flat with the three reduced regressions fitted by
``glm``, a configuration whose own consistency the concordance marks ``unverified``.  So the
study tested Theorem 1's conclusion without establishing its premise, and the cheapest way to
tell a falsified *configuration* from a falsified *estimator* is to refit both cells with the
reductions at their population limits and see whether the column moves.

**E2 ran and did not decide, and this module is the repair.**  Three cells of four came back
``unresolved``, every one on the same clause of gate B: *no other rung may be strictly better
than the shipped reference*.  A coarser rung beat ``spline(16)`` on one or another reduced
regression in three cells and lost to it in the fourth, which is §8's own falsifier and says
the reference's resolution is **not one choice**.  So the rung is now selected against a
measured ranking rather than shipped, and five things follow.  Each is a change to the
*reference* or an **addition** to the gates, each makes a verdict harder to reach, and none of
them touches :data:`EQUIVALENCE_FRACTION`, :data:`BUDGET_FRACTION` or
:data:`PRIMARY_ESTIMAND` -- the three constants the comparison is judged by, which were frozen
before E2 ran and are unchanged.

*Four blocks, because selecting and certifying are two jobs.*  E2 had three -- fit, score,
evaluate -- and three sufficed because it never selected: it shipped a rung and used the
scoring block to check it.  Turn that ranking into a selection and the same block does both,
and a rung certified by the block that chose it is a rung that certified itself.  So the
scoring block splits into a **selection** block and an **audit** block on disjoint scramble
streams, and the run is two passes over the draws with a barrier between them: the control arm
first, whose exit state supplies a ranking no rung's own fit produced (:class:`RecordingDRTMLE`
says why that state and not another); then the reference arm, at the rungs pass one selected,
audited on rows the selection never saw.

*And the selection is judged by the gate's own statistic*, which is what
:func:`select_rung` is: the coarsest rung that no other rung *significantly* beats, on paired
intervals over draws, exactly as gate B's second clause reads them.  A rule judged on point
estimates cannot be certified by a clause read on intervals -- measured, not argued, and the
docstring there carries the pilot that measured it.

*Five metrics, because none of the three regressions is what the fit reads.*
:func:`~cleverly.inference.influence.reduced_correction_parts` divides:
``H_2 = g_{r,2}/g_{r,1}`` and ``H_3 = q_r/g``, per arm, at the bounded denominators.  Those two
are ranked as losses in their own right beside the three componentwise ones, at the same
``g_bounds`` the fit used, with each divisor's margin and truncation rate recorded beside them
(:data:`~benchmarks.drtmle_reference.METRICS`).  Componentwise risks are theorem-relevant and
**incomplete**, not wrong.

*The negative control has to be detectably inferior on every metric it is meant to test.*  A
gate that cannot reject a deliberately coarse arm cannot certify anything, and at ``g-drift``
``2,400`` E2's ``bins(8)`` was not rejected on ``gr2`` -- so ``bins(8)`` is not coarse enough
to be a control there.  :data:`NEGATIVE_CONTROL` is coarser and its rejection is a gate clause
on all five metrics rather than on three.

*The block size is a lever and its falsifier is declared in advance.*  Gate B ranks candidates
at a **fixed** number of reference points, so "``spline(8)`` beats ``spline(16)``" is a
bias--variance statement about that block as much as about the ladder.  If doubling
``--reference-points`` moves the ranking towards the finer rungs, the winner was a statement
about the block and not about the function.

**This module runs that comparison, and it runs the gates that have to pass first.**
``benchmarks/drtmle_reference.py`` builds the reference and says in its own docstring what it
does not establish: the result is a **numerical reference and not an oracle**, and what is left
after the construction is a smoothing bias and a finite point count.  A paired
reference-against-``glm`` number read before those are bounded is not evidence about the
reduction learner, and this module exists so that it cannot be read that way -- the gate table
prints first, and a cell whose gates did not pass reads ``unresolved`` however large its
difference is.

**The rule the tables are read against is frozen above and in**
``docs/drtmle/validation-plan.md`` **§8**, in a commit that precedes every number: an
equivalence margin of a quarter of the ``glm`` arm's own level, a reference-uncertainty budget
of a third of that margin, and three verdicts of which ``unresolved`` is one and is not a weak
``equivalent``.  It may be changed before a dispatch with a written reason and not after one,
which is the discipline C3c's own value rests on.

**Why the gates cannot be a refinement difference, which is the trap here.**  The obvious
fidelity statistic is the movement of the reference between two knot counts.  That is the
statistic ``docs/roadmap.md`` withdrew for the quadrature ladder and then again for the binned
branches, and E2 inheriting it would rebuild the mistake it exists to repair.  Measured on this
repository's own ladder a successive difference ran four times *below* the true error at the
finest rung and three orders *above* it two rungs earlier; it is uninformative in both
directions, and it is undefined for the rung a ladder starts on.

**So the three gates are three different instruments, and none of them is that one.**

*A -- the exact-law control*, which is a test rather than a column and lives in
``tests/unit/test_reference_exact_law.py``: on a law whose conditioning index is discrete the
reduced regressions are finite sums, and the provider reproduces them array for array.  It
gates the **construction** -- the weights, the mask, the arms, the recomputation at the current
targeted pair -- and it is silent about the smoother, since no exact law has a continuum in it.

*B -- the held-out weighted risk*, here, and now on **five metrics and two blocks**.  Each
candidate is fitted on the reference block and scored on an **independent, finer** block from a
disjoint scramble stream: the *selection* block in pass one, which chooses the rung, and the
*audit* block in pass two, which certifies it and did not choose it.  The cross term vanishes
identically because a reference is a weighted :math:`L_2` projection
(:func:`~benchmarks.drtmle_reference.held_out_risk` carries the algebra) -- and it goes on
vanishing under a composite metric's weight, because both divisors are functions of the
conditioning index (:func:`~benchmarks.drtmle_reference.metric_weights` carries that half).  So
a *difference* of two candidates' risks estimates a difference of squared weighted errors on
every one of the five.  It **orients**, which is what a refinement difference cannot do: a
near-interpolating reference has a smaller movement and a larger risk, so the two point
opposite ways and only one of them ranks.

*C -- the randomisation budget*, here.  Independent scrambles of the **reference** block, one
refit each.  This is what a fixed grid's error cannot be estimated from and what E1b's device
supplies: a randomised quasi-Monte Carlo rule is unbiased at every point count, so the spread
across scrambles is a standard error assuming no rate.  Unlike E1b's evaluation rule it is
**not free** -- the reference enters the fit, so a scramble is a fit -- which is why the budget
runs on a subset of draws and is a declared cost rather than a default.

**What C cannot see, and B is why it is not alone.**  Every scramble shares the knot count and
the basis, so the across-scramble spread is orthogonal to a bias in the basis.  Randomisation
gates the quadrature half and the held-out risk gates the smoothing half; neither substitutes
for the other, and the exact-law control gates neither.

**The comparison is paired at every level it can be.**  One draw, one fold seed, one companion:
the ``glm`` arm and the reference arm see the same rows, the same prescribed nuisance
functions, the same split, and the **same evaluation windows and truths** -- so the evaluation
rule's own error, which E1b measured as very nearly the whole of a one-replicate study's
across-draw spread at :math:`n = 2{,}400`, is largely common to the two and cancels in the
difference.  What is left is the reduction learner.  **The two passes do not break that**: a
draw's seeds and companion layout are a function of its :class:`Payload` alone, so the arm
fitted in pass two sees the rows the arm fitted in pass one saw, and
``tests/unit/test_drtmle_reference_study.py`` pins it rather than leaving it to the reader.

**What this module does not do**, each refused by name.

* **No coverage claim.**  The reduction learner's effect on an interval is not this question,
  and a module that also reported a coverage number would be answering something its inputs
  were not frozen for.
* **No learner comparison.**  Growing-basis against ``glm`` against ``boost`` is E2b's and
  fires only if this branches.  The ladder here is the *reference's own* resolution, read as a
  fidelity gate rather than as a contest between estimator families.
* **No rate.**  Item 13 is a rate and closes at E5.  Two sizes here size the comparison, they
  do not carry an exponent.
* **The word oracle.**  The reference is a numerical estimate of a population limit and every
  gate above exists because it is one.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from cleverly import DRTMLE
from cleverly.estimators.base import format_table
from cleverly.utils.parallel import map_parallel

# The benchmarks package is a checkout rather than an installed distribution, so a plain
# `python benchmarks/drtmle_reference_study.py` has to find its siblings the way its
# neighbours do.
if __package__ in (None, ""):  # pragma: no cover - only on the direct-script path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import drtmle_injection, drtmle_remainder, drtmle_tier2
from benchmarks.drtmle_reference import (
    KNOT_LADDER,
    METRICS,
    EqualCountBins,
    ReferenceReductionDRTMLE,
    SplineProjection,
    arm_truth,
    composite_denominators,
    fit_mask,
    fold_targets,
    held_out_risk,
    metric_weights,
)

#: The two tiers, keyed as every harness on this page keys them so one interface serves both.
TIERS = {1: drtmle_injection, 2: drtmle_tier2}

#: Selected at import and replaced by ``main``; the tier is a module of designs rather than a
#: branch, exactly as it is in the coverage harness.
injection: Any = drtmle_injection

#: The learner the reference is compared **against**, and it is not a knob.  It is C3c's
#: configuration, which is the thing under test: the question is whether that configuration's
#: reductions are why the column is flat, and changing it would answer a different one.
REDUCED_LEARNER = "glm"

#: The rungs the selection chooses among and the audit ranks the winner against.  A ladder over
#: an integer, so there is no continuous constant to commit to, and it is read as a **risk
#: ordering** and never as a movement between two resolutions.
#:
#: **In ascending parameter count, which is the tie-break** -- :func:`select_rung` takes the
#: first minimum, so the simplest rung wins a tie exactly as
#: ``docs/roadmap.md`` says the simplest learner does.
RUNGS = tuple(SplineProjection(knots) for knots in sorted(KNOT_LADDER))

#: The rung the reference falls back to where a cell selected nothing -- a debug run with too
#: few draws to rank, or a pass-one failure.  E2's shipped middle rung, so a fallback is
#: visible as *E2's* choice in the record rather than as a selection nobody made.
FALLBACK_RUNG = SplineProjection(KNOT_LADDER[1])

#: The negative control gate B must reject, **on every metric**.  Not a candidate: it is here so
#: that "the gate rejects a reference that is too coarse" is a measurement rather than a hope,
#: and because a deliberately coarse arm reaches a *close final estimate* while being a badly
#: wrong function -- the one shape of result that would otherwise tempt a reader to skip the
#: gate.
#:
#: **Two bins rather than E2's eight, and this is a change to the instrument with a measurement
#: behind it rather than a preference.**  At ``g-drift`` ``2,400`` E2 read ``bins(8)`` against
#: ``spline(16)`` on ``gr2`` at ``-1.031e-04 [-4.35e-04, +1.12e-04]`` -- the point estimate the
#: *wrong* way and the interval straddling zero -- so the gate had no teeth there.  A control
#: that a well-chosen spline cannot be shown to beat is not a coarse arm, it is a second
#: candidate.
#:
#: A six-draw sandbox pilot in that same cell then measured how far the bin count has to fall,
#: and the answer was further than the obvious halving: ``bins(4)`` is rejected on ``qr``,
#: ``gr1``, ``gr2`` and ``h2`` -- including the ``gr2`` case E2 failed on, at
#: ``+2.076e-04 [+8.70e-05, +3.67e-04]`` -- and **not** on ``h3``, at
#: ``+7.190e-05 [-1.16e-04, +2.21e-04]``.  That metric divides by a *bound-active* targeted
#: mechanism, so its weight has the heavy tail :math:`1/g^{*2}_b` gives it, and a control has to
#: be wrong by more than a factor of two's worth of bins to clear it.
#:
#: The two coarser arms stay in :data:`REPORTED_CONTROLS` as ranked rows, which is where the
#: gate's actual discrimination is read: a clause on the coarsest control is a *necessary*
#: condition on the instrument, and it is the rung-against-rung intervals beside it -- differences
#: of ``1e-06`` resolved at six draws -- that say how finely the audit can tell two references
#: apart.
NEGATIVE_CONTROL = EqualCountBins(2)

#: Ranked and printed, gating nothing.  E2's own control and the halving between, kept so that
#: the run says how much discrimination the gate has rather than only that it has some, and so
#: that E2's column stays readable beside the repaired one.
REPORTED_CONTROLS = (EqualCountBins(4), EqualCountBins(8))

#: The four scramble streams, disjoint from each other and from every other study on this page.
#: Which block a rule takes must not depend on which rows a draw was fitted on; the selection's
#: must not meet the audit's, or the block that chose the rung is the block that certifies it;
#: and the reference's must not meet the evaluation's for a stronger reason still -- the
#: reference's error propagates into the fit **deterministically**, so a shared scramble would
#: make the fit and the integral the same random variable with a covariance nobody can sign.
#:
#: **These are fresh, and that is a decision rather than housekeeping.**  PR #74 is demoted to a
#: pilot and its seed streams are spent: a decision run on the streams that produced the pilot's
#: winners would be a run whose selection had already seen its own rows.
REFERENCE_SEED = 103_000_000
SELECTION_SEED = 104_000_000
AUDIT_SEED = 105_000_000
EVALUATION_SEED = 106_000_000

#: Sobol points per block.  The reference block is what the three regressions are fitted on and
#: is sized by the **points-per-parameter budget** the reference refuses to be fitted thinner
#: than: ``spline(32)`` has 35 parameters and so needs 2,240 rows, and ``qr``'s ``| A = a``
#: mask keeps about half of a block's ``2 * points``.  The selection and audit blocks are
#: deliberately finer, since a held-out risk carries its own Monte Carlo error and nothing pairs
#: it away -- and the audit's clause that failed E2 was a clause about *power*.
#:
#: **8,192 rather than E2's 4,096, and this is the lever pulled on its own falsifier's answer.**
#: ``docs/drtmle/validation-plan.md`` §8 declares it: gate B ranks at a *fixed* block size, so a
#: rung winning is a bias--variance statement about the block as much as about the ladder, and
#: *if doubling the block moves the ranking towards the finer rungs, the winner was a statement
#: about the block*.  A twelve-draw sandbox pilot ran both sizes on the same seeds and the
#: falsifier **did not fire**: the selection is rung for rung identical at 4,096 and 8,192, so the
#: winners are statements about the function.  What the doubling *did* do is resolve the audit --
#: at 4,096 the componentwise ``qr`` difference between ``spline(8)`` and ``spline(16)`` is a
#: **resolved** ``-7.6e-07`` and fails the gate, and at 8,192 it is ``-3.0e-07`` and straddles
#: zero, which is two rungs that are genuinely indistinguishable rather than one beating the
#: other.  It also takes ``h2``'s divisor back inside its bounds -- margin ``+0.1636`` against
#: ``-1.1693`` -- so the composite's weight stops carrying a two-order tail.  And it is nearly
#: free at four blocks: ``52.2s`` a fit against ``45.8s``, because a fit's cost is now the
#: companion's predictions rather than the reference's own least squares.
DEFAULT_REFERENCE_POINTS = 8_192
DEFAULT_SELECTION_POINTS = 8_192
DEFAULT_AUDIT_POINTS = 8_192
DEFAULT_EVALUATION_POINTS = 2_048

#: Independent evaluation scrambles averaged into one reading per fit.  Two rather than one
#: because :func:`~benchmarks.drtmle_remainder.remainder_rows` then reports a replicate
#: standard error rather than ``nan``, and because averaging a randomised rule halves its
#: variance -- both arms read the same ones, so this sharpens the *pair* and not one side.
DEFAULT_EVALUATION_SCRAMBLES = 2

#: Gate C's budget: independent reference scrambles, and the draws they are taken on.  A
#: refit each, unlike E1b's evaluation replicates, so this is a cost rather than a default and
#: the two are separate knobs.
DEFAULT_BUDGET_SCRAMBLES = 4
DEFAULT_BUDGET_DRAWS = 4

DEFAULT_SIZES = (600, 2_400)
DEFAULT_DRAWS = 32

#: Resamples behind every reported interval, and the percentiles they are read at.  A bootstrap
#: over **draws**, because the draw is the independent unit: a fit's gate rows and its
#: evaluation replicates are both conditional on it.
BOOTSTRAP = 2_000
INTERVAL = (2.5, 97.5)

#: Which reduced regression names are reported, in the order every table prints them.
REDUCTIONS = ("qr", "gr1", "gr2")

#: Which metrics each reduced regression is selected and audited on: its own componentwise risk
#: and the composite risk of whichever correction term divides by it.  ``gr1`` has one, because
#: it is a *divisor* rather than a numerator -- ``H_2``'s denominator is what it is.
METRICS_OF = {
    name: tuple(metric.name for metric in METRICS if metric.reduction == name)
    for name in REDUCTIONS
}


# ------------------------------------------------- the rule, frozen before the dispatch
#
# ``docs/drtmle/validation-plan.md`` §8 is this rule in prose, with what would falsify each
# half of it and the three readings it refuses.  The constants live here so that a run's
# banner and the record cannot come apart, and they may be changed before a dispatch with a
# written reason and not after one -- the clause §4, §5 and §7 all carry.

#: The equivalence margin, as a fraction of the ``glm`` arm's **own** level of
#: :math:`\sqrt n R_{\text{remaining}}` in the same cell and size.
#:
#: A fraction rather than an absolute number, and the reason is that the alternative is worse
#: rather than that this is ideal.  An absolute margin would have to be committed at one cell
#: and one size -- C3c's ``q-drift`` column reads ``1.43 / 1.26 / 1.25`` and ``g-drift``'s does
#: not fall at all -- and would then be arbitrary everywhere else.  A quarter of the level the
#: column already sits at is the statement candidate 1 actually makes: *the reductions are why
#: the remainder does not vanish*, so replacing them with their population limits should remove
#: a substantial share of it, and a quarter is where "substantial" is committed.
#:
#: **What is frozen is the fraction**, which is in this commit and predates every number the
#: comparison produces.  The level it scales is the run's own control arm, which is a
#: measurement -- said out loud here rather than left for a reader to notice.
EQUIVALENCE_FRACTION = 0.25

#: Gate C's budget: the reference's own across-scramble spread may be at most this share of the
#: margin.  A third, because a rule whose instrument's error is a comparable fraction of its own
#: decision boundary is a rule that decides on the instrument -- and because two thirds is the
#: point at which a single scramble's draw would move a verdict across the band on its own.
#: Exceeding it makes every verdict in that cell ``unresolved``, which is a statement about the
#: reference's resolution and is repaired by ``--reference-points`` rather than by a rerun.
BUDGET_FRACTION = 1.0 / 3.0

#: The estimand the verdict is read on, with the other two reported and **supporting**.
#: Declared in advance for the reason ``validation-plan.md`` §7 declares its own primary clause:
#: three estimands and two cells and two sizes is twelve readings, and a piece that chose which
#: of them to lead with after seeing them would be choosing its own conclusion.  The ATE is
#: primary because it is the contrast the demonstration is about; the two arm means carry
#: different drift coefficients and are read for whether one of them moved where the contrast
#: did not.
PRIMARY_ESTIMAND = "ate"

#: The three verdicts.  ``unresolved`` is a **third** verdict and is not a weak ``equivalent``:
#: it says the run cannot tell the two apart at this precision, which is a statement about the
#: study and not about the estimator, and the response to it is a larger draw count or a finer
#: reference rather than a conclusion.
VERDICTS = ("moved", "equivalent", "unresolved")


# ---------------------------------------------------- the selection rule, and it is E2R's own
#
# New with E2R, and a change to the **reference** rather than to the rule above: what rung the
# three reduced regressions are fitted at stops being shipped and starts being measured. It is
# declared here, in the commit that precedes the dispatch, for the reason every constant above
# is -- and the audit that certifies it reads a block this rule never saw.


def beaten_by(
    risks: Mapping[str, Mapping[str, Mapping[int, float]]], label: str
) -> list[tuple[str, str]]:
    """Which ``(metric, rung)`` pairs are **significantly** better than ``label``.

    The gate's own arithmetic, applied to the selection block: a paired per-draw difference and a
    bootstrap interval over draws, and ``other`` beats ``label`` when that interval lies wholly
    below zero.  One function, so the selection cannot be judged by a statistic the gate does not
    use -- which is the mistake this replaced.
    """
    out: list[tuple[str, str]] = []
    for metric, by_candidate in risks.items():
        base = by_candidate.get(label, {})
        for rung in RUNGS:
            if rung.label == label:
                continue
            draws = by_candidate.get(rung.label, {})
            shared = sorted(set(draws) & set(base))
            if len(shared) < 2:
                continue
            values = np.array([draws[seed] - base[seed] for seed in shared], dtype=float)
            if interval(values)[1] < 0.0:
                out.append((metric, rung.label))
    return out


def select_rung(risks: Mapping[str, Mapping[str, Mapping[int, float]]]) -> str:
    r"""Which rung one reduced regression is fitted at: the **coarsest unbeaten** one.

    ``risks`` is ``{metric: {candidate: {draw: risk}}}`` on the selection block, over the metrics
    that regression appears in -- its own componentwise risk, and the composite risk of the
    correction term that divides by it.

    **The rule is admissibility against the gate's own statistic, and that is not the rule this
    module first shipped.**  The first one minimised the worst *relative excess* of the mean
    risks, which reads as the natural minimax and is judged by a quantity the gate does not use.
    A six-draw pilot then showed what that costs: on ``qr`` at ``g-drift`` ``2,400`` it bought a
    ``0.002`` relative loss on the componentwise metric -- a difference of ``2e-06``, **resolved**
    at six draws, interval clear of zero -- with a ``0.01`` apparent gain on ``h3``, whose
    intervals at that precision straddle zero by an order of magnitude.  So it selected a rung
    that gate B then rejected, on a trade a reader would have called sensible.  The lesson
    generalises past this run: *a selection judged on point estimates cannot be certified by a
    clause read on intervals.*

    So a rung is **admissible** when no other rung's paired interval lies wholly below zero on any
    of its metrics -- exactly gate B's second clause, read on the block that chose rather than on
    the block that certifies.  The selected rung is the **coarsest** admissible one, since
    :data:`RUNGS` ascends in parameter count and a resolution the ladder cannot distinguish should
    be the one carrying the smaller variance and the weaker claim.  The audit then asks whether
    that admissibility **replicates** on rows the selection never saw, which is a far more useful
    question than whether a point-estimate winner survives an interval.

    Where no rung is admissible -- every one significantly beaten somewhere, which is a ladder
    with no unbeaten member rather than a tie -- the fewest-beaten rung is taken and the count
    goes on the record, so a cell in that state is visible rather than silently resolved.

    A metric with no reading contributes nothing rather than counting as agreement, and a
    regression with no reading at all raises: a selection that silently became a default is the
    mistake this whole rule replaces.
    """
    readable = {
        rung.label
        for by_candidate in risks.values()
        for rung in RUNGS
        if rung.label in by_candidate
    }
    if not readable:
        raise ValueError(
            "no rung has a reading on any metric, so nothing was selected. A run this thin "
            "cannot certify a reference; raise --draws rather than letting a default stand in"
        )
    counts = {
        rung.label: len(beaten_by(risks, rung.label)) for rung in RUNGS if rung.label in readable
    }
    fewest = min(counts.values())
    return next(rung.label for rung in RUNGS if counts.get(rung.label) == fewest)


def relative_excess(means: Mapping[str, float]) -> dict[str, float]:
    """Each rung's mean risk above the best rung's, as a fraction of the best rung's.

    Over the **rungs alone**: the negative control is not a candidate, and letting it into the
    denominator of a relative excess would make every rung's number a statement about how bad
    the control is.
    """
    readings = {
        label: float(value)
        for label, value in means.items()
        if np.isfinite(value) and label in {rung.label for rung in RUNGS}
    }
    if not readings:
        return {}
    best = min(readings.values())
    scale = abs(best) if abs(best) > 0.0 else 1.0
    return {label: (value - best) / scale for label, value in readings.items()}


# ------------------------------------------------------------------------------- the records


@dataclass
class FitRow:
    """One estimand of one fit, and the grain the paired table is arithmetic on.

    Flat and JSON-serialisable, as every record on this page is: a nested one would need a
    schema before a reader could recompute anything from the artefact.

    Attributes
    ----------
    estimator:
        ``"glm"`` or ``"reference"``.  The pairing key is ``(cell, n, data_seed, estimand)``,
        and the two arms share a fold seed and a companion.
    scramble:
        Which reference block this fit's reductions were fitted on.  ``0`` on the ``glm`` arm,
        which has no reference block -- and not the scramble it *would* have used, since a
        column that reads like a randomisation on an arm that took none is how a budget comes
        to be computed over the wrong rows.
    root_n_remaining:
        Item 13's column, averaged over this fit's evaluation replicates.  The paired
        difference of the two arms' values is the whole of what E2 decides on.
    companion_replicate_se:
        The spread of that column across those replicates, which is the evaluation rule's own
        error at this fit and is **not** the estimator's.
    companion_rows:
        Rows of the whole stacked companion this fit carried -- the reference blocks, the
        scoring block and the evaluation blocks together.  What a fit is *billed* in, since
        every one of them is a prediction per fold per nuisance, and it differs between an
        ordinary draw and a gate-C budget draw.
    """

    cell: str
    n: int
    data_seed: int
    fold_seed: int
    estimator: str
    scramble: int
    estimand: str
    psi: float
    truth: float
    p0_curve: float
    pn_curve: float
    remaining: float
    root_n_remaining: float
    companion_replicate_se: float
    companion_rows: int
    rounds: int
    exit_reason: str
    valid: bool
    seconds: float
    error: str = ""


@dataclass
class RiskRow:
    """Gate B's grain: one candidate's held-out risk on one **metric**, arm and fold.

    Reported apart rather than averaged, because the five metrics have different targets on
    different scales -- ``qr``'s is a residual and ``gr1``'s is an indicator, and a composite's
    is either divided by a mechanism -- so a mean over them would be a number with no units and
    a gate on it would be a gate on whichever is largest.

    Attributes
    ----------
    phase:
        ``"select"`` or ``"audit"``.  Which block this risk was scored on, and therefore which
        job it did: the first chooses the rung and the second certifies it.  One field rather
        than two files, because the two are the same measurement on different rows and a reader
        comparing them is the point.
    metric:
        One of :data:`~benchmarks.drtmle_reference.METRICS`' names.  ``qr``, ``gr1`` and ``gr2``
        are the componentwise risks; ``h3`` and ``h2`` are the same regressions scored where the
        estimator divides, which is what the fit is actually sensitive to.
    reduction:
        Which regression was fitted and scored.  Two metrics share it -- ``qr`` and ``h3``, and
        ``gr2`` and ``h2`` -- so a table keyed on the metric alone cannot say which of the three
        a row is about.
    divisor_margin, divisor_truncated:
        The composite's denominator, as
        :attr:`~benchmarks.drtmle_reference.Denominator.margin` and its truncation share; ``nan``
        on a componentwise row, which has no divisor.  Recorded beside the risk because a
        composite loss at a bound-active divisor is a loss whose weight the truncation chose.
    """

    cell: str
    n: int
    data_seed: int
    phase: str
    candidate: str
    metric: str
    reduction: str
    treatment_arm: float
    fold: int
    risk: float
    fitted_rows: int
    scored_rows: int
    divisor_margin: float = float("nan")
    divisor_truncated: float = float("nan")
    error: str = ""


@dataclass
class SelectionRow:
    """What pass one selected, per cell, size and reduced regression, and what chose it.

    The record of the repair.  Every number the comparison later reports is conditional on
    these three rungs, so an artefact that carried the comparison and not the selection would
    be a comparison at a reference nobody could name -- which is the reading E2 was held to and
    the reason this is a third file rather than a printed line.

    ``beaten`` is how many ``(metric, rung)`` pairs are *significantly* better than the winner on
    the selection block, which is :func:`select_rung`'s own objective: **zero** says the ladder has
    an unbeaten member and the audit's job is to see whether that replicates, and anything else
    says it does not and the count is on the record rather than resolved silently.

    ``excess`` is the winner's relative excess risk on its worst metric and ``runner_up`` the next
    rung's -- reported rather than selected on, since a point estimate is not what the gate reads,
    and kept because it says whether the choice was a contest or a formality.
    """

    cell: str
    n: int
    reduction: str
    selected: str
    beaten: int
    excess: float
    runner_up: str
    runner_up_excess: float
    metrics: str
    draws: int


@dataclass(frozen=True)
class Payload:
    """One draw: both arms, its gate rows, and however many reference scrambles it carries.

    ``rungs`` is what pass one selected, as ``(reduction, knots)`` pairs -- a plain value rather
    than the :class:`~benchmarks.drtmle_reference.SplineProjection` objects it stands for,
    because a payload crosses a process boundary and a record has to be readable off the
    artefact without importing this module.  Empty in pass one, which has nothing selected yet.
    """

    cell: str
    n: int
    data_seed: int
    fold_seed: int
    reference_points: int
    selection_points: int
    audit_points: int
    evaluation_points: int
    evaluation_scrambles: int
    reference_scrambles: int
    rungs: tuple[tuple[str, int], ...] = ()

    def references(self) -> dict[str, Any]:
        """The rungs as a per-regression reference mapping, or E2's rung where none was chosen.

        A fallback that is *visible* -- :data:`FALLBACK_RUNG` is E2's shipped middle rung and the
        record says so -- rather than a silent default, which is the mistake E2R exists to
        remove.
        """
        chosen = {name: SplineProjection(knots) for name, knots in self.rungs}
        return {name: chosen.get(name, FALLBACK_RUNG) for name in REDUCTIONS}


@dataclass(frozen=True)
class Layout:
    """A draw's companion, sliced into the four roles its blocks play.

    One frame and one weight vector for all of them, which is
    :func:`~benchmarks.drtmle_remainder.stacked_companion`'s contract and is what stops a
    caller pairing a window with the wrong rule's measure.  Two things about it are properties
    of the *roles* rather than of the rule.  The blocks are at **different resolutions**: a
    scored block has to be finer than the block a candidate was fitted on, since a held-out
    risk carries its own error and nothing pairs it away.  And **selection and audit are two
    blocks**, on disjoint scramble streams, because a rung certified by the block that chose it
    is a rung that certified itself.
    """

    stack: Any
    reference: tuple[Any, ...]
    selection: Any
    audit: Any
    evaluation: tuple[Any, ...]


def layout(payload: Payload, dgp: Any) -> Layout:
    """Every block one draw needs, in one companion, from four disjoint scramble streams.

    A function of the :class:`Payload` alone, which is what makes the two passes a pair: the
    control arm fitted in pass one and the reference arm fitted in pass two see the same rows,
    the same windows and the same truths, so the evaluation rule's own error is common to them
    and cancels in the paired difference.  ``reference_scrambles`` is the one field that differs
    between an ordinary draw and a gate-C budget draw, and it only ever *adds* reference blocks
    ahead of the other three -- so a budget draw's windows are not an ordinary draw's, and that
    is why a draw is fitted at one layout in both passes rather than at whichever one it needs.
    """
    offset = payload.data_seed % 1_000_003
    reference_seeds = [REFERENCE_SEED + offset + i for i in range(payload.reference_scrambles)]
    evaluation_seeds = [EVALUATION_SEED + offset + i for i in range(payload.evaluation_scrambles)]
    scrambles = [
        *reference_seeds,
        SELECTION_SEED + offset,
        AUDIT_SEED + offset,
        *evaluation_seeds,
    ]
    points = [
        *[payload.reference_points] * len(reference_seeds),
        payload.selection_points,
        payload.audit_points,
        *[payload.evaluation_points] * len(evaluation_seeds),
    ]
    stack = drtmle_remainder.stacked_companion(dgp, points=points, scrambles=scrambles)
    taken = len(reference_seeds)
    return Layout(
        stack=stack,
        reference=stack.blocks[:taken],
        selection=stack.blocks[taken],
        audit=stack.blocks[taken + 1],
        evaluation=stack.blocks[taken + 2 :],
    )


class RecordingDRTMLE(DRTMLE):
    """The control arm, keeping the last state its own reduction refit was handed.

    **This is the state the rungs are ranked at, and the choice is E2R's and not an accident.**
    A rung has to be chosen at *some* state, because the state fixes the conditioning index every
    candidate is a regression on, and the audit reads the alternation's **exit**.  Three states
    were available and two of them are wrong here:

    * a *reference arm's* exit state is the self-certification the four blocks exist to prevent,
      arrived at through the state instead of through the rows: the rung would be chosen at the
      exit of a fit that already used a rung;
    * the **initial pair** is candidate-free and was this module's first choice, and the same
      six-draw pilot that sized :data:`NEGATIVE_CONTROL` is why it is not the one kept: at the
      initial pair every rung's ``h3`` reading favoured the coarsest, and at the exit state the
      audit's ``h3`` point estimates favoured a finer one.  ``h3`` divides by :math:`g^*_b`, whose
      bound-activity is *made* by targeting -- the pilot read its margin at ``-0.0135`` -- so a
      ranking taken before the alternation is a ranking at a divisor the fit never uses;
    * the **control arm's** exit state is candidate-free -- no rung was involved in producing it,
      the ``glm`` reduction is the thing under comparison rather than a candidate in the ladder --
      and it is a *targeted* state, so the selection and the audit read the same kind of divisor.

    The residue is stated rather than hidden: the reference arm's own exit state is not the
    control arm's, because different reductions target differently.  No selection can be made at
    the state it will be certified at without certifying itself, which is why the audit exists.
    """

    state: Any = None
    bounds: tuple[float, float] | None = None
    produced: tuple[Any, ...] = ()

    def _reduction(self, data: Any, nuisance: Any) -> Any:
        spec = super()._reduction(data, nuisance)
        if spec is None:  # pragma: no cover - a DRTMLE fit always carries one
            return None
        inner = spec.refit
        bounds = nuisance.reduced.g_bounds

        def refit(current: Any) -> Any:
            self.state, self.bounds = current, bounds
            result = inner(current)
            self.produced = tuple(result[1])
            return result

        return replace(spec, refit=refit)


class RecordingReferenceDRTMLE(ReferenceReductionDRTMLE):
    """Keeps the last nuisance state its provider was handed, **and what it produced there**.

    Gate B has to score candidates **at a state the reference actually answered at**, and the
    alternation's last one is the one the reported reductions came from.  Nothing on a
    ``TMLEResult`` carries it: ``nuisance`` is the *initial* pair, deliberately, because
    ``NuisanceEstimates.outcome`` stays the initial regression exactly as it does on ``TMLE``.
    Recording it here is a few lines and keeps the gate a statement about the fit rather than
    about its starting point.

    ``produced`` is the companion reduction the provider returned at that state, and it is
    recorded for one reason: ``h2``'s divisor is :math:`g_{r,1,b}`, and the copy the *state*
    carries is one refit behind the arrays the reported correction was built from.  A weight one
    round stale would not invalidate a ranking -- it is common to every candidate either way --
    but it would make the recorded margin and truncation share describe an array the fit did not
    divide by, and those two columns are read as item 25's.
    """

    state: Any = None
    bounds: tuple[float, float] | None = None
    produced: tuple[Any, ...] = ()

    def _reference_set(self, current: Any, g_bounds: tuple[float, float]) -> Any:
        self.state = current
        self.bounds = g_bounds
        result = super()._reference_set(current, g_bounds)
        self.produced = tuple(result[1])
        return result


# ------------------------------------------------------------------------------ one draw


def _fit(payload: Payload, estimator: Any) -> Any:
    import warnings

    dgp = injection.base_law()
    frame, _ = dgp.sample(payload.n, seed=payload.data_seed)
    with warnings.catch_warnings():
        # Positivity warnings are per-draw noise here, as they are in the coverage harness.
        warnings.simplefilter("ignore")
        return estimator.fit(frame, outcome="Y", treatment="A").single()


def _settings(payload: Payload, evaluation: Any) -> dict[str, Any]:
    return dict(
        injection.settings(payload.cell, payload.n),
        reduced_outcome_learner=REDUCED_LEARNER,
        reduced_treatment_learner=REDUCED_LEARNER,
        random_state=payload.fold_seed,
        evaluation=evaluation,
    )


def _alternation(fit: Any) -> dict[str, Any]:
    reduction = fit.repeats[0].fluctuations["mean"].reduction
    if reduction is None:  # pragma: no cover - a DRTMLE fit always carries one
        return {"rounds": 0, "exit_reason": ""}
    return {"rounds": int(reduction.rounds), "exit_reason": str(reduction.exit_reason)}


def _fit_rows(
    payload: Payload,
    fit: Any,
    place: Layout,
    *,
    estimator: str,
    scramble: int,
    seconds: float,
) -> list[FitRow]:
    """Every estimand's remainder columns at one fit, read on the evaluation blocks.

    Both arms are handed the *same* windows and the same per-window :math:`\\psi_0`, which is
    what makes the difference below a paired quantity rather than two independent readings of
    an integral.  Each window carries its own truth because the cancellation
    :func:`~benchmarks.drtmle_remainder.truth_at` documents is **within** a replicate.
    """
    dgp = injection.base_law()
    check = fit.validation.score_check()
    alternation = _alternation(fit)
    windows = [block.window for block in place.evaluation]
    truths = [
        drtmle_remainder.truth_at(dgp, block.points, scramble=block.seed)
        for block in place.evaluation
    ]
    measured = drtmle_remainder.remainder_rows(
        fit,
        dgp,
        n=payload.n,
        bounds=fit.config.g_bounds,
        row_weights=place.stack.weights,
        windows=windows,
        truths=truths,
    )
    return [
        FitRow(
            cell=payload.cell,
            n=payload.n,
            data_seed=payload.data_seed,
            fold_seed=payload.fold_seed,
            estimator=estimator,
            scramble=scramble,
            estimand=row.estimand,
            psi=row.psi,
            truth=row.truth,
            p0_curve=row.p0_curve,
            pn_curve=row.pn_curve,
            remaining=row.remaining,
            root_n_remaining=row.root_n_remaining,
            companion_replicate_se=row.companion_replicate_se,
            companion_rows=int(place.stack.weights.size),
            valid=bool(check.passed),
            seconds=seconds,
            **alternation,
        )
        for row in measured
    ]


def _failed_fit(
    payload: Payload, estimator: str, scramble: int, error: str, rows: int
) -> list[FitRow]:
    nan = float("nan")
    return [
        FitRow(
            cell=payload.cell,
            n=payload.n,
            data_seed=payload.data_seed,
            fold_seed=payload.fold_seed,
            estimator=estimator,
            scramble=scramble,
            estimand=name,
            psi=nan,
            truth=nan,
            p0_curve=nan,
            pn_curve=nan,
            remaining=nan,
            root_n_remaining=nan,
            companion_replicate_se=nan,
            companion_rows=rows,
            rounds=0,
            exit_reason="",
            valid=False,
            seconds=nan,
            error=error,
        )
        for name in ("ate", "ey1", "ey0")
    ]


def risk_rows(
    payload: Payload,
    state: Any,
    bounds: Any,
    place: Layout,
    *,
    phase: str,
    reduced: Sequence[Any] | None = None,
) -> list[RiskRow]:
    r"""Gate B: every candidate's held-out weighted risk, per **metric**, arm and fold.

    Fitted on the reference block and scored on the block ``phase`` names -- the selection block
    in pass one and the audit block in pass two, each a **different scramble at a finer
    resolution**, from streams disjoint from each other, from the reference's and from the
    evaluation's.  One function for both, because a selection and a certification that were two
    pieces of arithmetic could differ in a way no table would show; what differs between the two
    calls is the window and nothing else.

    The mask travels with the regression: ``qr``'s ``| A = a`` restricts the rows it is scored
    on exactly as it restricts the rows it is fitted on, or the risk would be taken under a
    measure the regression is not defined at.  **The metric's weight travels with the metric**:
    a composite is the same fit and the same rows under :math:`w/d^2`, so a candidate is fitted
    once per ``(reduction, arm, fold)`` and scored once per metric that reads it.  Fitting it
    twice would be the same arithmetic and would invite the two from coming apart.

    ``reduced`` is the companion reduction ``h2``'s divisor is read from, defaulting to the one
    ``state`` carries -- see :attr:`RecordingReferenceDRTMLE.produced` for when it is passed.

    A candidate whose points-per-parameter budget the block does not meet is recorded as a row
    carrying its refusal rather than dropped: a rung that could not be scored is a gap in the
    gate and has to look like one.
    """
    dgp = injection.base_law()
    mass = np.asarray(place.stack.weights, dtype=float)
    fitting = _block_mask(place.reference[0].window, mass.size)
    block = place.selection if phase == "select" else place.audit
    scoring = _block_mask(block.window, mass.size)
    candidates = (*RUNGS, NEGATIVE_CONTROL, *REPORTED_CONTROLS)

    rows: list[RiskRow] = []
    for arm in state.arms:
        truth = arm_truth(state, dgp=dgp, arm=arm)
        for fold in range(state.companion.n_folds):
            designs, targets = fold_targets(state, fold=fold, arm=arm, truth=truth, g_bounds=bounds)
            divisors = composite_denominators(
                state,
                fold=fold,
                arm=arm,
                g_bounds=bounds,
                reduced=None if reduced is None else reduced[fold],
            )
            weights = metric_weights(mass, divisors)
            for name in REDUCTIONS:
                keep = fit_mask(name, truth.indicator)
                inside = fitting if keep is None else (fitting & keep)
                outside = scoring if keep is None else (scoring & keep)
                for candidate in candidates:
                    try:
                        fitted: Any = candidate.fit(
                            designs[name][inside], targets[name][inside], mass[inside]
                        )
                        refusal = ""
                    except Exception as exc:  # recorded, never swallowed
                        fitted, refusal = None, type(exc).__name__
                    for metric in METRICS:
                        if metric.reduction != name:
                            continue
                        divisor = None if metric.divisor is None else divisors[metric.divisor]
                        row = RiskRow(
                            cell=payload.cell,
                            n=payload.n,
                            data_seed=payload.data_seed,
                            phase=phase,
                            candidate=candidate.label,
                            metric=metric.name,
                            reduction=name,
                            treatment_arm=float(arm),
                            fold=fold,
                            risk=float("nan"),
                            fitted_rows=int(inside.sum()),
                            scored_rows=int(outside.sum()),
                            divisor_margin=float("nan") if divisor is None else divisor.margin,
                            divisor_truncated=(
                                float("nan") if divisor is None else divisor.truncated
                            ),
                            error=refusal,
                        )
                        if fitted is not None:
                            try:
                                row.risk = held_out_risk(
                                    fitted,
                                    designs[name][outside],
                                    targets[name][outside],
                                    weights[metric.name][outside],
                                )
                            except Exception as exc:  # recorded, never swallowed
                                row.error = type(exc).__name__
                        rows.append(row)
    return rows


def _block_mask(window: Any, rows: int) -> np.ndarray:
    mask = np.zeros(rows, dtype=bool)
    mask[window.start : window.stop] = True
    return mask


def control_draw(payload: Payload) -> tuple[list[FitRow], list[RiskRow]]:
    """**Pass one**: the ``glm`` arm, and the ranking the rung is selected from.

    The control arm is fitted once whatever the budget is -- it has no reference block, so a
    scramble does nothing to it, and refitting it per scramble would price a control that cannot
    move.

    **The ranking is taken at this fit's exit state**, for the reasons
    :class:`RecordingDRTMLE`'s docstring gives: it is candidate-free, and it is targeted, so the
    selection reads the same kind of divisor the audit will.  ``h2``'s divisor is therefore the
    **control arm's own** ``gr1`` and ``h3``'s its own targeted mechanism -- the arrays *this* fit
    divides by, common to every candidate the ranking compares, which is the only property the
    ranking needs of them.
    """
    dgp = injection.base_law()
    place = layout(payload, dgp)

    rows: list[FitRow] = []
    risks: list[RiskRow] = []

    started = time.perf_counter()
    estimator = RecordingDRTMLE(**_settings(payload, place.stack.frame))
    try:
        plain = _fit(payload, estimator)
    except Exception as exc:  # recorded, never swallowed
        return (
            _failed_fit(payload, "glm", 0, type(exc).__name__, place.stack.weights.size),
            risks,
        )
    rows.extend(
        _fit_rows(
            payload,
            plain,
            place,
            estimator="glm",
            scramble=0,
            seconds=time.perf_counter() - started,
        )
    )
    try:
        risks.extend(
            risk_rows(
                payload,
                estimator.state if estimator.state is not None else plain.nuisance,
                estimator.bounds or plain.config.g_bounds,
                place,
                phase="select",
                reduced=estimator.produced or None,
            )
        )
    except Exception as exc:  # pragma: no cover - recorded, never hidden
        print(f"the selection ranking is unavailable on {payload.cell} n={payload.n}: {exc!r}")
    return rows, risks


def reference_draw(payload: Payload) -> tuple[list[FitRow], list[RiskRow]]:
    """**Pass two**: the reference arm at the selected rungs, and the audit that certifies them.

    One fit per reference scramble, the extra ones being gate C's budget.  The audit is read
    once per draw, off the **first** reference arm's recorded state and the reduction it produced
    there -- the state the reported correction was built at, on a block the selection never saw.
    """
    dgp = injection.base_law()
    place = layout(payload, dgp)
    references = payload.references()

    rows: list[FitRow] = []
    risks: list[RiskRow] = []

    for index, block in enumerate(place.reference):
        started = time.perf_counter()
        try:
            estimator = RecordingReferenceDRTMLE(
                dgp=dgp,
                reference=references,
                window=block.window,
                row_weights=place.stack.weights,
                **_settings(payload, place.stack.frame),
            )
            fit = _fit(payload, estimator)
        except Exception as exc:  # recorded, never swallowed
            rows.extend(
                _failed_fit(
                    payload,
                    "reference",
                    block.seed,
                    type(exc).__name__,
                    place.stack.weights.size,
                )
            )
            continue
        rows.extend(
            _fit_rows(
                payload,
                fit,
                place,
                estimator="reference",
                scramble=block.seed,
                seconds=time.perf_counter() - started,
            )
        )
        if index == 0 and estimator.state is not None:
            try:
                risks.extend(
                    risk_rows(
                        payload,
                        estimator.state,
                        estimator.bounds,
                        place,
                        phase="audit",
                        reduced=estimator.produced or None,
                    )
                )
            except Exception as exc:  # pragma: no cover - recorded, never hidden
                print(f"gate B unavailable on {payload.cell} n={payload.n}: {exc!r}")
    return rows, risks


# ------------------------------------------------------------------------------- the tables


def _finite(values: Sequence[float]) -> np.ndarray:
    return np.array([v for v in values if np.isfinite(v)], dtype=float)


def _mean(values: Sequence[float]) -> float:
    finite = _finite(values)
    return float(finite.mean()) if finite.size else float("nan")


def _cells(rows: Sequence[Any]) -> list[tuple[str, int]]:
    return sorted({(row.cell, row.n) for row in rows})


def paired_differences(rows: Sequence[FitRow], cell: str, n: int, estimand: str) -> np.ndarray:
    r"""``root_n_remaining`` at the reference minus at ``glm``, one value per draw.

    Paired on ``data_seed``, and on the **first** reference scramble where a draw carries
    several: gate C's extra scrambles are a budget measurement and folding them into the
    comparison would give a budget draw several times the weight of an ordinary one.
    """

    def select(estimator: str) -> dict[int, float]:
        picked: dict[int, tuple[int, float]] = {}
        for row in rows:
            if (row.cell, row.n, row.estimand, row.estimator) != (cell, n, estimand, estimator):
                continue
            if row.error or not np.isfinite(row.root_n_remaining):
                continue
            seen = picked.get(row.data_seed)
            if seen is None or row.scramble < seen[0]:
                picked[row.data_seed] = (row.scramble, row.root_n_remaining)
        return {seed: value for seed, (_, value) in picked.items()}

    plain, reference = select("glm"), select("reference")
    shared = sorted(set(plain) & set(reference))
    return np.array([reference[seed] - plain[seed] for seed in shared], dtype=float)


def budget_spread(rows: Sequence[FitRow], cell: str, n: int, estimand: str) -> tuple[float, int]:
    r"""Gate C: the reference's own across-scramble sd of the column, and the draws behind it.

    The **within-draw** spread averaged over draws, which is the reference's error *at a fixed
    fit* -- the same decomposition E1b takes for the evaluation rule, and it is a standard
    error under no assumption about a convergence rate because a randomised quasi-Monte Carlo
    rule is unbiased at every point count.

    What it cannot see is the smoothing bias: every scramble shares the knot count and the
    basis, so this spread is orthogonal to a bias in the basis.  Gate B is that half.
    """
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if (row.cell, row.n, row.estimand, row.estimator) != (cell, n, estimand, "reference"):
            continue
        if row.error or not np.isfinite(row.root_n_remaining):
            continue
        grouped.setdefault(row.data_seed, []).append(row.root_n_remaining)
    within = [float(np.var(np.array(v), ddof=1)) for v in grouped.values() if len(v) > 1]
    if not within:
        return (float("nan"), 0)
    return (float(np.sqrt(np.mean(within))), len(within))


def draw_risks(
    rows: Sequence[RiskRow], cell: str, n: int, metric: str, phase: str
) -> dict[str, dict[int, float]]:
    """``{candidate: {draw: mean risk}}`` on one metric, averaged over ``(arm, fold)``.

    Averaged **within** a draw, because those are the same regression problem at different
    splits rather than replicates of one number.  The draw is the independent unit and every
    interval below resamples it.
    """
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        if (row.cell, row.n, row.metric, row.phase) != (cell, n, metric, phase) or row.error:
            continue
        if not np.isfinite(row.risk):
            continue
        grouped.setdefault(row.candidate, {}).setdefault(row.data_seed, []).append(row.risk)
    return {
        label: {seed: float(np.mean(values)) for seed, values in draws.items()}
        for label, draws in grouped.items()
    }


def risk_gaps(
    rows: Sequence[RiskRow], cell: str, n: int, metric: str, *, phase: str, baseline: str
) -> dict[str, np.ndarray]:
    """Each candidate's risk minus the selected rung's, one paired value per draw.

    Paired against the selected rung's own risk on the same rows and the same draw, which is
    what makes the difference a difference of squared weighted errors rather than of two noisy
    risks.  ``baseline`` is what pass one selected for this metric's *reduction*; a metric
    with no baseline reading returns an empty mapping rather than falling back to another
    candidate, since a gap against a rung nobody selected answers for the wrong reference.
    """
    means = draw_risks(rows, cell, n, metric, phase)
    base = means.get(baseline, {})
    return {
        label: np.array(
            [draws[seed] - base[seed] for seed in sorted(set(draws) & set(base))], dtype=float
        )
        for label, draws in means.items()
        if label != baseline
    }


# ------------------------------------------------------------------- what pass one selected


def selection_rows(rows: Sequence[RiskRow]) -> list[SelectionRow]:
    """:func:`select_rung`'s verdict per cell, size and reduced regression, with its runner-up.

    Read off the ``select`` phase alone.  The audit rows are the same arithmetic on a different
    block and reading them here is exactly the self-certification the split exists to prevent --
    which is why ``phase`` is a field rather than a comment.
    """
    out: list[SelectionRow] = []
    picked = [row for row in rows if row.phase == "select"]
    for cell, n in _cells(picked):
        for reduction in REDUCTIONS:
            metrics = METRICS_OF[reduction]
            risks: dict[str, dict[str, dict[int, float]]] = {}
            excess: dict[str, dict[str, float]] = {}
            draws = 0
            for metric in metrics:
                means = draw_risks(picked, cell, n, metric, "select")
                if not means:
                    continue
                draws = max(draws, max((len(v) for v in means.values()), default=0))
                risks[metric] = means
                pooled = {label: _mean(list(values.values())) for label, values in means.items()}
                readings = relative_excess(pooled)
                if readings:
                    excess[metric] = readings
            if not risks:
                continue
            worst = {
                rung.label: max(
                    (
                        readings[rung.label]
                        for readings in excess.values()
                        if rung.label in readings
                    ),
                    default=float("nan"),
                )
                for rung in RUNGS
            }
            selected = select_rung(risks)
            others = sorted(
                (value, label)
                for label, value in worst.items()
                if label != selected and np.isfinite(value)
            )
            out.append(
                SelectionRow(
                    cell=cell,
                    n=n,
                    reduction=reduction,
                    selected=selected,
                    beaten=len(beaten_by(risks, selected)),
                    excess=float(worst.get(selected, float("nan"))),
                    runner_up=others[0][1] if others else "-",
                    runner_up_excess=others[0][0] if others else float("nan"),
                    metrics=" ".join(metrics),
                    draws=draws,
                )
            )
    return out


def selected_rungs(rows: Sequence[SelectionRow]) -> dict[tuple[str, int], dict[str, str]]:
    """``{(cell, n): {reduction: label}}`` -- the mapping pass two fits at and the audit reads."""
    out: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        out.setdefault((row.cell, row.n), {})[row.reduction] = row.selected
    return out


def selected_knots(
    rows: Sequence[SelectionRow],
) -> dict[tuple[str, int], tuple[tuple[str, int], ...]]:
    """The same mapping as :class:`Payload`'s ``rungs``: ``(reduction, knots)`` pairs.

    The label is mapped back to a knot count **here alone**, and only for labels this module's
    own :data:`RUNGS` produced -- a label from anywhere else is a rung nothing selected, so it
    raises rather than being coerced into one.
    """
    knots = {rung.label: rung.n_knots for rung in RUNGS}
    gathered: dict[tuple[str, int], dict[str, int]] = {}
    for row in rows:
        if row.selected not in knots:
            raise ValueError(
                f"{row.selected!r} is not one of this run's rungs {sorted(knots)}; a selection "
                "outside the declared ladder is a rung nobody committed to"
            )
        gathered.setdefault((row.cell, row.n), {})[row.reduction] = knots[row.selected]
    return {key: tuple(sorted(value.items())) for key, value in gathered.items()}


def interval(values: np.ndarray, seed: int = 20250801) -> tuple[float, float]:
    """A percentile interval for a mean, resampling **draws** -- the independent unit here."""
    if values.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.size, (BOOTSTRAP, values.size))
    means = values[picks].mean(axis=1)
    low, high = np.percentile(means, INTERVAL)
    return (float(low), float(high))


def equivalence_margin(rows: Sequence[FitRow], cell: str, n: int, estimand: str) -> float:
    r""":data:`EQUIVALENCE_FRACTION` times the ``glm`` arm's own level of the column.

    Read on the *paired* draws rather than on every ``glm`` row, so the margin and the
    difference it judges are computed over one set of draws.  A margin taken over draws the
    reference arm failed on would judge a comparison against a level no comparison was made at.
    """
    paired = {
        row.data_seed
        for row in rows
        if (row.cell, row.n, row.estimand, row.estimator) == (cell, n, estimand, "reference")
        and not row.error
        and np.isfinite(row.root_n_remaining)
    }
    level = _mean(
        [
            row.root_n_remaining
            for row in rows
            if (row.cell, row.n, row.estimand, row.estimator) == (cell, n, estimand, "glm")
            and row.data_seed in paired
            and not row.error
        ]
    )
    return float(EQUIVALENCE_FRACTION * abs(level))


def gate_verdict(
    fits: Sequence[FitRow],
    risks: Sequence[RiskRow],
    cell: str,
    n: int,
    chosen: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """``(verdict, reason)`` for the two measured gates in one cell.

    ``pass`` only if both hold.  Anything else is ``fail``, and a failed gate makes every
    comparison verdict in the cell ``unresolved`` -- which is a statement about the reference
    and not about the estimator, and is repaired by a finer reference or a larger draw count
    rather than by reading the comparison anyway.

    **B is read on the audit block and on all five metrics**, and it has two clauses of which
    the second is the one that is easy to leave out.  The negative control must be *rejected* --
    its held-out risk strictly worse at the interval's lower end -- or the gate has no teeth and
    cannot discriminate at this draw count; and no other rung may be strictly *better* at the
    interval's upper end, because a comparison run at a reference another resolution beats is a
    comparison answering for the wrong reference.

    **What E2R changes about B is which reference those clauses are about.**  E2 shipped a rung
    and this clause caught it; here the rung is what pass one selected, the clauses are read on
    a block the selection never saw, and they hold the *selection* to replicating out of sample.
    So a failure now says something different from E2's: not "the shipped rung was not the best
    one" but "the rung that won on one block is beaten on another", which is a statement about
    how well resolved the ladder is at this block size -- and
    ``docs/drtmle/validation-plan.md`` §8's fourth E2R clause is the lever for it.

    ``chosen`` is ``{reduction: label}`` from pass one.  Without it every clause is read against
    :data:`FALLBACK_RUNG`, which is what a debug run with no selection is doing and is recorded
    as such rather than silently.

    **C is one clause and is unchanged**: the reference's own across-scramble spread against
    :data:`BUDGET_FRACTION` of the margin, read at the primary estimand.
    """
    reasons: list[str] = []
    picked = dict(chosen or {})
    for metric in METRICS:
        baseline = picked.get(metric.reduction, FALLBACK_RUNG.label)
        gaps = risk_gaps(risks, cell, n, metric.name, phase="audit", baseline=baseline)
        control = gaps.get(NEGATIVE_CONTROL.label)
        if control is None or control.size < 2:
            reasons.append(f"B: {metric.name} has no control reading")
            continue
        if interval(control)[0] <= 0.0:
            reasons.append(f"B: {NEGATIVE_CONTROL.label} not rejected on {metric.name}")
        others = {rung.label for rung in RUNGS} - {baseline}
        for label in sorted(others):
            values = gaps.get(label, np.array([], dtype=float))
            # A rung that could not be scored is a **gap in the gate** and has to fail rather
            # than pass by absence: it is most often the points-per-parameter budget refusing
            # the finest rung on a block too thin to carry it, which means the ladder the
            # selection ranged over was shorter than the one this run declared.
            if values.size < 2:
                reasons.append(f"B: {label} has no reading on {metric.name}")
                continue
            if interval(values)[1] < 0.0:
                reasons.append(f"B: {label} beats {baseline} on {metric.name}")

    spread, draws = budget_spread(fits, cell, n, PRIMARY_ESTIMAND)
    margin = equivalence_margin(fits, cell, n, PRIMARY_ESTIMAND)
    if not np.isfinite(spread) or draws == 0:
        reasons.append("C: no budget draw")
    elif not (spread <= BUDGET_FRACTION * margin):
        reasons.append(f"C: sd {spread:.4f} over {BUDGET_FRACTION * margin:.4f}")
    return ("pass", "") if not reasons else ("fail", "; ".join(reasons))


def comparison_verdict(
    fits: Sequence[FitRow],
    risks: Sequence[RiskRow],
    cell: str,
    n: int,
    estimand: str,
    chosen: Mapping[str, str] | None = None,
) -> str:
    r"""``moved`` / ``equivalent`` / ``unresolved``, against the frozen margin.

    The rule ``docs/drtmle/validation-plan.md`` §8 states, in code so that a table cannot say
    something the rule does not:

    * **moved** -- the paired interval lies wholly **outside** :math:`[-\delta, +\delta]`.  The
      reduction learner materially changes item 13's column, candidate 1 is alive, and E2b
      fires.  Movement in *either* direction counts: a reference that makes the remainder
      larger is still a learner effect, and it is a finding E2b would have to explain rather
      than a null result.
    * **equivalent** -- the interval lies wholly **inside** it.  Candidate 1 is dead, the
      learner road is shut, and the diagnosis moves to E3.
    * **unresolved** -- anything else, and every cell whose gates did not pass.
    """
    if gate_verdict(fits, risks, cell, n, chosen)[0] != "pass":
        return "unresolved"
    differences = paired_differences(fits, cell, n, estimand)
    if differences.size < 2:
        return "unresolved"
    low, high = interval(differences)
    margin = equivalence_margin(fits, cell, n, estimand)
    if not np.isfinite(margin) or margin <= 0.0:
        return "unresolved"
    if low > margin or high < -margin:
        return "moved"
    if -margin <= low and high <= margin:
        return "equivalent"
    return "unresolved"


#: Headers for :func:`selection_table`, declared beside it -- the same hazard every harness here
#: guards against, and pinned the same way.
SELECTION_HEADERS = (
    "cell",
    "n",
    "reduction",
    "metrics",
    "selected",
    "beaten on",
    "worst excess",
    "runner-up",
)


def selection_table(rows: Sequence[SelectionRow]) -> list[list[str]]:
    """What pass one chose, and by how much -- printed **before** the gates.

    Read as four things at once.  It is the record of which rung each reduced regression is
    fitted at, without which every number below is a comparison at a reference nobody can name.
    ``beaten on`` is the rule's own objective and the column to read second: **`0`** says the
    ladder had an unbeaten member on the selection block, so the audit is asking whether that
    replicates; anything else says it did not, and the cell is one where no resolution in the
    ladder is admissible.  It is the answer to the question E2's falsifier raised -- *does one
    rung serve both cells* -- which is read off whether the ``selected`` column is constant down
    it and **not** off any verdict.  And ``worst excess`` beside ``runner-up`` says whether the
    choice was a contest, reported rather than selected on: two rungs within a per cent of each
    other is a ladder that cannot tell them apart on point estimates, which is a different
    statement from either being beaten.
    """
    return [
        [
            row.cell,
            f"{row.n:,}",
            row.reduction,
            row.metrics,
            row.selected,
            str(row.beaten),
            f"{row.excess:+.4f}",
            f"{row.runner_up} at {row.runner_up_excess:+.4f}"
            if np.isfinite(row.runner_up_excess)
            else "-",
        ]
        for row in rows
    ]


#: Headers for :func:`gate_rows`, declared beside it -- the same hazard every harness here
#: guards against, and pinned the same way.
GATE_HEADERS = ("gate", "cell", "n", "reading", "draws", "verdict")


def gate_rows(
    fits: Sequence[FitRow],
    risks: Sequence[RiskRow],
    chosen: Mapping[tuple[str, int], Mapping[str, str]] | None = None,
) -> list[list[str]]:
    """Gates B and C, printed **before** any paired number, with the cell's verdict on each.

    Gate A is not here: it is an exact-law control and a test, so it has already either passed
    or turned the suite red before a dispatch could start.

    Every reading is a *difference* with an interval, never a ratio.  A held-out risk carries
    the irreducible ``E_0[w(T - m)^2]`` of its own target, which is common to every candidate
    and can dominate both -- so a ratio of two risks is near one whatever the candidates are,
    and only the difference estimates a difference of squared weighted errors.

    **The ``audit`` rows are the gate and the ``select`` rows are printed beside them**, on the
    same five metrics.  The pair is worth the width: a rung that wins on the selection block and
    is beaten on the audit block is the one shape of result the four-block split exists to make
    visible, and a table carrying only the clause that failed would leave a reader unable to
    tell it from a rung that was never ahead.

    ``B. divisor`` is the composite denominators' own columns -- the margin and the truncation
    share of :math:`g^*_b` and :math:`g_{r,1,b}` -- which say whether a composite loss was taken
    at a *bound-active* divisor.  Reported and not gated: a bound-active fit is not a failing
    fit, exactly as ``cleverly.validation.drtmle`` says of item 25's own label.

    The ``verdict`` column is the **cell's**, repeated on each of its rows, and the reason is
    that the two gates fail together: a comparison behind either is ``unresolved``, so a
    per-row pass would invite a reader to take the rows that passed.
    """
    rows: list[list[str]] = []
    picked = dict(chosen or {})
    verdicts = {
        (cell, n): gate_verdict(fits, risks, cell, n, picked.get((cell, n)))
        for cell, n in sorted(set(_cells(risks)) | set(_cells(fits)))
    }
    for cell, n in _cells(risks):
        for metric in METRICS:
            baseline = picked.get((cell, n), {}).get(metric.reduction, FALLBACK_RUNG.label)
            for phase in ("audit", "select"):
                gaps = risk_gaps(risks, cell, n, metric.name, phase=phase, baseline=baseline)
                for label in sorted(gaps):
                    values = gaps[label]
                    low, high = interval(values)
                    rows.append(
                        [
                            f"B. {phase} vs {label}",
                            cell,
                            f"{n:,}",
                            f"{_mean(values):+.3e} [{low:+.2e}, {high:+.2e}] on {metric.name} "
                            f"(vs {baseline})",
                            str(values.size),
                            verdicts[(cell, n)][0] if phase == "audit" else "-",
                        ]
                    )
        for divisor in ("h3", "h2"):
            here = [
                row
                for row in risks
                if (row.cell, row.n, row.metric, row.phase) == (cell, n, divisor, "audit")
            ]
            rows.append(
                [
                    f"B. divisor {divisor}",
                    cell,
                    f"{n:,}",
                    f"margin {min((row.divisor_margin for row in here), default=float('nan')):+.4f}"
                    f", truncated {_mean([row.divisor_truncated for row in here]):.4f}",
                    str(len({row.data_seed for row in here})),
                    "-",
                ]
            )
    for cell, n in _cells(fits):
        spread, draws = budget_spread(fits, cell, n, PRIMARY_ESTIMAND)
        allowed = BUDGET_FRACTION * equivalence_margin(fits, cell, n, PRIMARY_ESTIMAND)
        rows.append(
            [
                "C. reference sd",
                cell,
                f"{n:,}",
                f"{spread:.4f} against {allowed:.4f}" if np.isfinite(spread) else "not measured",
                str(draws),
                verdicts[(cell, n)][0],
            ]
        )
    for (cell, n), (verdict, reason) in sorted(verdicts.items()):
        if reason:
            rows.append(["why", cell, f"{n:,}", reason, "-", verdict])
    return rows


#: Headers for :func:`comparison_rows`, declared beside it.
COMPARISON_HEADERS = (
    "cell",
    "n",
    "estimand",
    "draws",
    "glm",
    "reference",
    "paired d",
    "d 95%",
    "margin",
    "rule se",
    "verdict",
)


def comparison_rows(
    rows: Sequence[FitRow],
    risks: Sequence[RiskRow],
    chosen: Mapping[tuple[str, int], Mapping[str, str]] | None = None,
) -> list[list[str]]:
    r"""The paired comparison against the frozen margin, with its verdict.

    ``glm`` and ``reference`` are the mean of :math:`\sqrt n R_{\text{remaining}}` in each arm,
    and ``paired d`` is the mean of the per-draw difference with a bootstrap interval over
    draws.  Read the paired column and not the two levels: the evaluation rule's own error is
    very nearly the whole of a one-replicate study's across-draw spread at :math:`n = 2{,}400`
    (E1b), it is common to the two arms, and it cancels in the difference and not in either
    level.

    ``rule se`` is the evaluation rule's error at a fit, averaged over fits -- the column that
    says how much of the two levels is the instrument.  It is *not* the error of ``paired d``,
    which is the interval beside it.

    The ATE row is the one the piece branches on; ``ey1`` and ``ey0`` are supporting and are
    printed because the two arm means carry different drift coefficients and a contrast can
    cancel a movement one of them made.  Both statements are frozen in
    :data:`PRIMARY_ESTIMAND` rather than chosen from this table.
    """
    out: list[list[str]] = []
    picked = dict(chosen or {})
    for cell, n in _cells(rows):
        for estimand in ("ate", "ey1", "ey0"):
            differences = paired_differences(rows, cell, n, estimand)
            low, high = interval(differences)
            margin = equivalence_margin(rows, cell, n, estimand)
            here = [
                row
                for row in rows
                if (row.cell, row.n, row.estimand) == (cell, n, estimand) and not row.error
            ]
            levels = {
                arm: _mean([row.root_n_remaining for row in here if row.estimator == arm])
                for arm in ("glm", "reference")
            }
            out.append(
                [
                    cell,
                    f"{n:,}",
                    estimand,
                    str(differences.size),
                    f"{levels['glm']:+.4f}",
                    f"{levels['reference']:+.4f}",
                    f"{_mean(differences):+.4f}" if differences.size else "-",
                    f"[{low:+.4f}, {high:+.4f}]" if differences.size > 1 else "-",
                    f"+/-{margin:.4f}" if np.isfinite(margin) else "-",
                    f"{_mean([row.companion_replicate_se for row in here]):.4f}",
                    comparison_verdict(rows, risks, cell, n, estimand, picked.get((cell, n))),
                ]
            )
    return out


#: Headers for :func:`cost_rows`, declared beside it.
COST_HEADERS = ("cell", "n", "companion rows", "fits", "secs/fit", "rounds", "invalid")


def cost_rows(rows: Sequence[FitRow]) -> list[list[str]]:
    """What the design costs, priced in **fits** rather than in draws.

    A draw buys one ``glm`` fit and one reference fit per reference scramble, so a budget draw
    is several times an ordinary one and a table that priced a draw would understate it.
    """
    out: list[list[str]] = []
    for cell, n in _cells(rows):
        here = [row for row in rows if (row.cell, row.n) == (cell, n) and row.estimand == "ate"]
        timed = [row for row in here if np.isfinite(row.seconds)]
        if not timed:
            continue
        out.append(
            [
                cell,
                f"{n:,}",
                f"{int(np.median([row.companion_rows for row in timed])):,}",
                str(len(timed)),
                f"{_mean([row.seconds for row in timed]):.2f}",
                f"{_mean([float(row.rounds) for row in timed]):.1f}",
                str(sum(1 for row in here if row.error or not row.valid)),
            ]
        )
    return out


def write_records(
    fits: Sequence[FitRow],
    risks: Sequence[RiskRow],
    picks: Sequence[SelectionRow],
    directory: Path,
) -> tuple[Path, ...]:
    """Three artefacts from one timestamp, so they join.

    The three are different grains -- one row per ``(fit, estimand)``, one per
    ``(phase, candidate, metric, arm, fold)`` and one per ``(cell, size, regression)`` -- and a
    single file at the coarsest would make the two tables above unrecomputable from the
    evidence, which is the whole reason the standing decision on manifested rows exists.

    **The selection file is new with E2R and is not a convenience.**  Every number the
    comparison reports is conditional on which rung each reduced regression was fitted at, so an
    artefact set without it would record a comparison at a reference nobody could name.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    paths = (
        directory / f"{stamp}.jsonl",
        directory / f"{stamp}-risks.jsonl",
        directory / f"{stamp}-selection.jsonl",
    )
    for path, records in zip(paths, (fits, risks, picks), strict=True):
        with path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record)) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        nargs="+",
        default=list(drtmle_injection.CELLS),
        choices=list(drtmle_injection.CELLS),
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument(
        "--reference-points",
        type=int,
        default=DEFAULT_REFERENCE_POINTS,
        help="Sobol points the reduced regressions are fitted on. Sized by the reference's "
        "own points-per-parameter budget, which it refuses to be fitted thinner than",
    )
    parser.add_argument(
        "--selection-points",
        type=int,
        default=DEFAULT_SELECTION_POINTS,
        help="Sobol points the rung is SELECTED on, from a disjoint scramble stream. Finer than "
        "the fitting block on purpose: a held-out risk carries its own error and nothing pairs "
        "it away",
    )
    parser.add_argument(
        "--audit-points",
        type=int,
        default=DEFAULT_AUDIT_POINTS,
        help="Sobol points gate B CERTIFIES on, from a stream disjoint from the selection's. A "
        "rung certified by the block that chose it is a rung that certified itself",
    )
    parser.add_argument("--evaluation-points", type=int, default=DEFAULT_EVALUATION_POINTS)
    parser.add_argument(
        "--evaluation-scrambles",
        type=int,
        default=DEFAULT_EVALUATION_SCRAMBLES,
        help="independent randomisations of the evaluation rule, averaged into one reading "
        "per fit and shared by both arms so the pairing removes them",
    )
    parser.add_argument(
        "--budget-scrambles",
        type=int,
        default=DEFAULT_BUDGET_SCRAMBLES,
        help="gate C: independent reference scrambles on a budget draw. Each is a REFIT, "
        "unlike an evaluation replicate, which is why it runs on a subset of draws",
    )
    parser.add_argument("--budget-draws", type=int, default=DEFAULT_BUDGET_DRAWS)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20250801)
    parser.add_argument("--tier", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/drtmle-reference"),
        help="where the per-fit and per-candidate JSONL go; git-ignored generated output",
    )
    args = parser.parse_args()

    global injection
    injection = TIERS[args.tier]

    drawn = np.random.SeedSequence(args.seed).generate_state(2 * args.draws)
    seeds = [
        (int(data), int(fold))
        for data, fold in zip(drawn[: args.draws], drawn[args.draws :], strict=True)
    ]
    payloads = [
        Payload(
            cell=cell,
            n=n,
            data_seed=data_seed,
            fold_seed=fold_seed,
            reference_points=args.reference_points,
            selection_points=args.selection_points,
            audit_points=args.audit_points,
            evaluation_points=args.evaluation_points,
            evaluation_scrambles=args.evaluation_scrambles,
            # The budget draws are the first few of the stream rather than a random subset, so
            # that raising `--draws` cannot change which draws carry it.
            reference_scrambles=args.budget_scrambles if index < args.budget_draws else 1,
        )
        for cell in args.cells
        for n in args.sizes
        for index, (data_seed, fold_seed) in enumerate(seeds)
    ]
    print(
        f"tier {args.tier}: {len(payloads)} draws over cells {list(args.cells)} and sizes "
        f"{list(args.sizes)}, rungs {[rung.label for rung in RUNGS]} selected per (cell, size, "
        f"regression) on {args.reference_points:,} fitting points against {REDUCED_LEARNER}, "
        f"chosen on {args.selection_points:,} and certified on {args.audit_points:,}, control "
        f"{NEGATIVE_CONTROL.label}, evaluated on {args.evaluation_scrambles} x "
        f"{args.evaluation_points:,}, budget {args.budget_scrambles} scramble(s) on "
        f"{args.budget_draws} draw(s), jobs={args.jobs}"
    )

    # Two passes with a barrier, which is the four-block split expressed in the control flow:
    # pass one fits the control arm and ranks the rungs at its INITIAL pair -- a state every rung
    # shares -- and pass two fits the reference arm at what that ranking selected. A single pass
    # could not do it: a rung selected inside the draw that used it would be selected at a state
    # its own fit produced.
    started = time.perf_counter()
    first = map_parallel(control_draw, [(payload,) for payload in payloads], n_jobs=args.jobs)
    fits = [row for batch, _ in first for row in batch]
    risks = [row for _, batch in first for row in batch]

    picks = selection_rows(risks)
    knots = selected_knots(picks)
    chosen = selected_rungs(picks)
    summary = "; ".join(
        f"{cell} n={n:,} " + " ".join(f"{name}={label}" for name, label in sorted(rungs.items()))
        for (cell, n), rungs in sorted(chosen.items())
    )
    print(f"\npass one selected: {summary}")

    selected = [
        (replace(payload, rungs=knots.get((payload.cell, payload.n), ())),) for payload in payloads
    ]
    second = map_parallel(reference_draw, selected, n_jobs=args.jobs)
    elapsed = time.perf_counter() - started
    fits += [row for batch, _ in second for row in batch]
    risks += [row for _, batch in second for row in batch]
    paths = write_records(fits, risks, picks, args.out)

    def table(title: str, headers: Sequence[str], rows: list[list[str]]) -> None:
        print(f"\n{title}")
        print("=" * len(title))
        print(format_table(list(headers), rows))

    table("What pass one selected", SELECTION_HEADERS, selection_table(picks))
    table(
        "The fidelity gates, and read these first",
        GATE_HEADERS,
        gate_rows(fits, risks, chosen),
    )
    table("The paired comparison", COMPARISON_HEADERS, comparison_rows(fits, risks, chosen))
    table("What it cost", COST_HEADERS, cost_rows(fits))

    failures = [row for row in fits if row.error]
    if failures:
        print(f"\n{len(failures)} fit(s) failed: {sorted({row.error for row in failures})}")

    print("\nReading the numbers")
    print("=" * 19)
    print(
        "Read the selection table first, because every number under it is conditional on\n"
        "which rung each reduced regression was fitted at. `worst excess` is the winner's\n"
        "own relative excess risk on its worst metric and `runner-up` is the next rung's:\n"
        "two rungs within a per cent of each other is a ladder that cannot tell them apart.\n"
        "Whether ONE rung serves both cells is read off whether the `selected` column is\n"
        "constant down this table, and off nothing else.\n"
        "\n"
        "Then the gate table, as differences with intervals. `B. audit vs` is a candidate's\n"
        "held-out weighted risk MINUS the SELECTED rung's, on rows neither the selection nor\n"
        "the candidate saw. A positive value says the selected rung is the better estimate;\n"
        "a negative one says another rung beats it, which fails the gate. `B. select vs` is\n"
        "the same difference on the block that chose, printed beside it so a rung that won\n"
        "there and lost here is visible rather than inferred. It is a difference and never\n"
        "a ratio, because a risk carries the irreducible variance of its own target and\n"
        "that part is common to every candidate.\n"
        "\n"
        "Five metrics, not three. `qr`, `gr1` and `gr2` are the componentwise risks; `h3`\n"
        "and `h2` are the same regressions scored where the fit's correction DIVIDES --\n"
        "q_r/g and g_r2/g_r1 at the bounded denominators -- which is what a fit is actually\n"
        "sensitive to. Componentwise risks are theorem-relevant and incomplete, not wrong.\n"
        "`B. divisor` carries each denominator's margin and truncation share, reported and\n"
        "not gated: a bound-active fit is not a failing fit.\n"
        "\n"
        "`C. reference sd` is the reference's own contribution to the column, measured by\n"
        "independent scrambles of the block it is fitted on -- one REFIT each. It is a\n"
        "standard error under no assumption about a rate, and it is blind to the smoothing\n"
        "bias, which every scramble shares. Gate B is that half and neither substitutes.\n"
        "\n"
        "`paired d` is the whole comparison: the reference's column minus `glm`'s, on the\n"
        "same draw, the same split and the same evaluation windows. Read it rather than the\n"
        "two levels beside it -- the evaluation rule's error is most of each level and\n"
        "cancels in the difference.\n"
        "\n"
        "`margin` is a QUARTER of the glm arm's own level, frozen as a fraction before any\n"
        "of these numbers existed (docs/drtmle/validation-plan.md section 8). `moved` is the\n"
        "interval wholly outside it -- candidate 1 alive, E2b fires; `equivalent` is wholly\n"
        "inside -- candidate 1 dead, the diagnosis moves to E3; `unresolved` is anything else\n"
        "and is a THIRD verdict, not a weak `equivalent`. A failed gate makes its whole cell\n"
        "unresolved, and the repair is a finer reference or more draws rather than reading\n"
        "the comparison anyway.\n"
        "\n"
        "The ATE row is the one the piece branches on and the two arm means are supporting;\n"
        "both were declared before the run rather than chosen from the table.\n"
        "\n"
        "Nothing here is a coverage claim, nothing here selects a LEARNER -- a rung of one\n"
        "deterministic basis is a resolution and not an estimator family, and the learner\n"
        "comparison is E2b's -- and nothing here reads a rate. Item 13 is a rate and closes\n"
        "at E5."
    )
    print(
        f"\n{len(fits)} fit row(s), {len(risks)} gate row(s) and {len(picks)} selection row(s) "
        f"in {elapsed:.0f}s wall clock."
    )
    for path in paths:
        print(f"Rows: {path}")


if __name__ == "__main__":
    main()
