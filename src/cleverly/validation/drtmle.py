r"""Is the curve a doubly-robust fit reports the one it solved for?

A :class:`~cleverly.DRTMLE` fit records three solved scores and then reports an influence
curve built from the arrays those solves left behind.  :mod:`cleverly.validation.score`
checks both ends of that -- the scores the targeting step recorded, and the mean of the
curve the estimate is built from.  What nothing checked until this module is that they are
the same statement about the same state, and they were not:

.. code-block:: text

    equation (9), as solved     Pn[ H_g (A - g*) ]        g* RAW,       from solve_mechanism
    D*_g, as reported           Qr/g-bar* (1_a - g-bar*)  g* TRUNCATED, from reduced_corrections

Both read one array.  Only one truncated it in the *residual*, and the covariate's
denominator was truncated in both -- so the two expressions were identical on every row the
bound left alone and differed on every row it clipped.  A single clipped row of 600 was
enough to leave the reported curve uncentred at ``2e-04`` while all three fluctuation rows
reported their scores solved to ``1e-11``. This module makes that mismatch impossible to hide: it
recomputes each correction's empirical mean **from the exact returned state**, compares it
with the score the solver recorded, and reports the discrepancy as :math:`\Delta_g` and
:math:`\Delta_Q` alongside the clipping bias :math:`B_{clip}` that explained it.

The bounded implementation replaces the solver at the ``DRTMLE`` call sites with
:func:`~cleverly.fluctuation.mechanism.solve_bounded_mechanism`, which solves the score at
the *truncated* tilt -- the expression the second line above carries -- and the alternation
now carries that truncated array forward.  The two lines are one line, :math:`\Delta` is at
roundoff on every fit, and :math:`B_{clip}` is zero because there is no longer a second array
for it to measure the distance to.  **Nothing here was weakened to make that true**: every
threshold, tolerance and condition below is what it was when the identity failed, which is
the only reason the rows are worth reading now.

Two failures, and they are not the same failure
-----------------------------------------------

*An identity residual* -- :math:`\Delta_g` or :math:`\Delta_Q` above roundoff -- is a
**software defect**.  The fit solved one expression and reported another, and no amount of
further iteration would fix it because the loop is not posing the equation the curve needs.

*A correction score* -- :math:`P_n[w D^*_g]` or :math:`P_n[w D^*_Q]` above the inferential
tolerance -- is a **fit that did not solve its equations**, which is the ordinary thing
:mod:`cleverly.validation.score` is about, reported per arm and per equation so that a
reader can see which one.

Both are reported through :class:`~cleverly.validation.score.ScoreCheck`, which is what
``summary()`` prints from, and each keeps its own wording there.

*And a row that is neither.*  A fit guarding one nuisance solves one of the two extra
equations, so its curve subtracts one correction -- and the other term is still reported
here, marked :attr:`~CorrectionRow.solved` ``False``, as the diagnostic saying what is
**not** in this curve.  Such a row is not a failure and cannot be one: nothing subtracts
it. This row also guards partial fits against subtracting a correction they did not solve for.
The instrument's first run against a ``guard=("g",)`` fit reported :math:`2.8\times10^{-3}`
at arm 1 with **no** row clipped and no equation (9) anywhere in the fit -- and at the time
the curve subtracted that term anyway.  It no longer does; the row that found it stays.

What this does **not** do is choose a convention, and it did not need to: every identity here
is valid under all four candidates, which is what let it land a piece ahead of the one that
chose.  Which mechanism equation (9) ought to be solved against was a derivation rather than
a taste -- the theorem's own algorithm truncates nothing at all, so there was no convention
in the source to match and no document to wait for. The reasoning is on
:func:`~cleverly.fluctuation.mechanism.solve_bounded_mechanism`.

Five conditions on how the identity is checked, each ruling out a way of passing for the
wrong reason, and all five are in the code below rather than in this docstring's good
intentions:

* **per arm**, never only on the ATE -- arm-specific errors cancel in a difference, and the
  ATE curve is the rowwise difference of the arm curves;
* **before** the contrast is constructed;
* with the **row weights** included, since every score here is weighted;
* on **one outcome scale**.  :math:`Q_r` and the fluctuation's residual live on the
  ``[0, 1]`` scaled outcome; the reported curve carries
  :attr:`~cleverly.utils.bounds.OutcomeScaler.range`.  Everything here is reported on the
  **outcome's own scale**, so that a correction score and ``se / sqrt(n)`` are comparable
  numbers rather than two quantities a factor of ``range`` apart -- which is the same trap
  in a second place;
* on a fixture where the truncation **binds**.  That one belongs to the tests, and the
  witness is :attr:`CorrectionRow.margin` -- **not** :attr:`CorrectionRow.clipped`, which the
  bounded state empties. The alternation carries the truncated array forward, so a converged fit
  clips
  nothing at the exit however hard the draw was, and a fixture selected on the exit count
  would now be selected from nothing at all. That would make the check agree where it could not
  have disagreed. What a constrained
  root does instead is sit *against* the boundary, and the margin is how that shows.

Which estimator the fit is, which is a different question
---------------------------------------------------------

Everything above asks whether a fit solved what it reports.  :attr:`CorrectionCheck.contract`
asks something no other column here answers: **which estimator the numbers are evidence
about.** Truncation is not in
Theorem 1's algorithm, so the theorem-backed guarantee is claimed for a fit whose truncations
are *inactive*, and a fit where one binds is reported as empirically supported and outside the
theorem.  **Three** truncations have to be inactive for that on a complete-data fit, not one,
and **five** on a randomized missing-outcome one -- which divides by two mechanisms, truncated
separately at two different bounds:

=========================  ========================================  ===================
truncation                 witness                                   in the assumptions
=========================  ========================================  ===================
:math:`\hat g` at the fit  :attr:`CorrectionRow.initial_clipped`      an assumption on
                                                                     :math:`g_0`, not an
                                                                     operation on
                                                                     :math:`\hat g`
:math:`\hat\pi` at the     :attr:`CorrectionRow.observation_clipped`  an assumption on
fit                                                                  :math:`\pi_0`, on
                                                                     the same footing
:math:`g^*` at the exit    :attr:`CorrectionRow.margin`               the same one
:math:`\pi^*` at the exit  :attr:`CorrectionRow.observation_margin`   the same one
:math:`g_{r,1}`, or        :attr:`CorrectionRow.gr1_margin`           **none**: it is a
:math:`\gamma_a` and                                                 regression of an arm
:math:`\gamma_\Delta` on                                             indicator on
a missing-outcome fit                                                :math:`\hat{\bar Q}`,
                                                                     and :math:`g_0 >
                                                                     \delta` does not
                                                                     imply it is bounded
                                                                     away from zero
=========================  ========================================  ===================

The two :math:`\pi` rows are on missing-outcome fits only; on a complete-data fit their
columns hold their sentinels and :attr:`CorrectionCheck.truncations_active` does not name
them.

**The two observation rows are not implied by the treatment ones.**  A randomized trial's
treatment mechanism is flat by design and cannot clip, while its observation mechanism is a
fitted probability that can sit at its floor on a large share of rows -- so a check reading
only the first three reports ``"theorem"`` on a fit that is squarely bound-active.

**A bound-active fit is not a failing fit, and :attr:`CorrectionCheck.passed` does not read
this.**  On ``weak_overlap_dgp`` every identity holds at ``1e-17`` and every score is
negligible while a third of the ``(row, arm)`` pairs clip at the initial mechanism; that is a
scope label, and folding it into a verdict would report the fit as broken.  The label needs a
threshold where the two margin columns deliberately have none, so the threshold is on the
label -- :data:`MARGIN_ACTIVE` -- and the raw columns stay what they were.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..estimators.tmle import correction_parts, reported_mechanism
from ..utils.frames import emit_frame
from ..utils.records import sentinel_equality
from ..utils.text import format_table

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult

__all__ = [
    "IDENTITY_TOLERANCE",
    "MARGIN_ACTIVE",
    "CorrectionCheck",
    "CorrectionRow",
    "correction_check",
]

#: How far a state identity may miss before it is called a defect, on the outcome's scale.
#: A number rather than a judgement, and the gap either side of it is what makes it one:
#: measured on ``nonlinear_dgp`` at ``n = 400``, the residual is ``2e-19`` on a draw where
#: the identity holds and ``1.3e-04`` on one where it does not, with the smallest observed
#: real failure at ``7e-08``.  So this sits seven orders above the arithmetic and four
#: below the defect, and it is deliberately *not* expressed relative to the score: the
#: quantity is a difference of two evaluations of one expression, and its right value is
#: zero rather than something small compared with anything.
IDENTITY_TOLERANCE = 1e-12

#: How near a bound a mechanism has to come before :attr:`CorrectionCheck.contract` calls the
#: truncation *active*, as a fraction of the interval between the bounds.  A threshold on the
#: **label** and not on the columns it reads, deliberately: :attr:`CorrectionRow.margin` and
#: :attr:`CorrectionRow.gr1_margin` stay thresholdless so a test can pick its own, and this is
#: the one place a number is needed, because a scope label has to come out of a comparison.
#:
#: The gap either side of it is what makes it a number rather than a judgement.  On the draw
#: the centring mismatch was found on the exit margin is ``1.2e-06`` and on its sibling it
#: is ``0.14``; over a 96-fit dispatch ``weak-overlap``'s median
#: margin is **exactly** ``0.0e+00`` at two of three sizes while the three ordinary processes
#: sit at ``0.11`` to ``0.20``.  So this sits two orders above the active regime and three below
#: the inactive one.
MARGIN_ACTIVE = 1e-4


@sentinel_equality
@dataclass(frozen=True)
class CorrectionRow:
    """One arm, one equation, one draw: what was solved against what is reported.

    Attributes
    ----------
    equation:
        ``"D*_g"`` -- equation (9), the mechanism tilt -- or ``"D*_Q"``, equation (10).
    stored:
        The score the targeting step recorded for this arm's column of that equation, or
        ``nan`` when this fit solved no such equation.  The record is *faithful*: it is a
        genuine evaluation of the expression the solver posed, which is why the identity
        below is a statement about two expressions rather than about two states.
    reported:
        :math:`P_n[w D^*]` for the same arm, recomputed from the returned state through the
        very expression the reported curve subtracts -- or, where :attr:`solved` is
        ``False``, would subtract if the fit had solved for it.  Built by
        :func:`~cleverly.inference.influence.reduced_correction_parts`, which forms both
        terms whatever the guard is precisely so this row exists.
    clip_bias:
        :math:`B_{clip}(a) = P_n[w\\,Q_r/g^b\\,(g - g^b)]` for the ``"D*_g"`` row, ``nan``
        for the other. Before bounded targeting it reproduced **minus** :attr:`residual` to
        floating point, which made it a check on the centring diagnosis rather than merely a new
        column; the sign is the orientation
        :attr:`~cleverly.inference.influence.CorrectionParts.clip_bias` defines it in,
        and the two differed because one residual read :math:`1_a - g` and the other
        :math:`1_a - g^b`.  It is **zero now**, on every fit, because there is no longer a
        raw tilted mechanism for it to measure the distance to.
    clipped:
        How many rows the truncation binds on *at the exit*. Zero on every converged bounded fit,
        and that is the point rather than a defect in the column: the alternation carries
        the truncated array forward, so at a fixed point :math:`\\epsilon \\to 0` and the raw
        and truncated tilts coincide.  What it says now is that the fix took.  Use
        :attr:`margin` to ask whether the draw was a hard one.
    margin:
        The closest the targeted **treatment** mechanism comes to either bound, as a
        fraction of the
        interval between them: ``min_i min(g*_i - lo, hi - g*_i) / (hi - lo)``.  This is the
        witness that replaces :attr:`clipped`, and it says what that column used to say --
        **whether the truncation had anything to do on this draw** -- in the only form that
        survives bounded targeting. A constrained root sits *at* the boundary of the feasible
        set, so a
        draw whose unconstrained tilt wanted to leave the bounds comes back with the
        mechanism pressed against one: measured at ``1.2e-06`` on the original failing draw
        on, against ``0.14`` on its sibling that never clipped.  Five orders, and no
        threshold inside the column -- a test picks its own.

        It is not a proof that the constraint was active, and nothing derivable from the
        returned arrays is: the trajectory that got there is not on the record.  What it is
        is a property that separates the two draws by five orders and cannot be manufactured
        by the fix, which is what a fixture's precondition has to be.
    initial_clipped:
        How many ``(row, arm)`` pairs of the **initial treatment** mechanism -- the
        cross-fitted
        :meth:`~cleverly.estimators._nuisance.Propensity.arm`, untruncated -- lie outside the
        truncation.  A property of the *draw* rather than of the alternation, and so the one
        truncation witness here that is not about what the loop did: it counts what
        :meth:`~cleverly.estimators._nuisance.Propensity.bounded` had to do on the way into
        equation (8)'s covariate. Unlike :attr:`clipped`, which the bounded state empties, this
        column
        **can** disagree -- a 96-fit dispatch read a share of ``0.000``
        on ``linear``, ``nonlinear`` and ``off-diagonal`` and ``0.231`` to ``0.338`` on
        ``weak-overlap``, separating the measured overlap regimes.
    gr1_margin:
        The closest :math:`g_{r,1}` comes to either bound, as a fraction of the interval,
        read off the **untruncated** array :class:`~cleverly.estimators.reduced.ReducedSet`
        stores -- so unlike :attr:`margin` it is **signed**, and a value at or below zero says
        :meth:`~cleverly.estimators.reduced.ReducedSet.bounded_gr1` is doing something to
        equation (10)'s denominator. It is the third relevant truncation and the one with no
        counterpart in Theorem 1's assumption list:
        :math:`g_{r,1}` is a regression of an arm indicator on :math:`\\hat{\\bar Q}`, and
        :math:`g_0 > \\delta` does not imply it is bounded away from zero.
        ``weak-overlap``'s ``min gr1`` of ``0.000`` is this bound binding.

        **On a missing-outcome fit this column carries something else**, because
        :math:`g_{r,1}` does not exist in that construction: it is the smaller of
        :math:`\\gamma_a`'s margin against ``g_bounds`` and :math:`\\gamma_\\Delta`'s
        against ``[nuisance_bound, 1]``.  The column is reused rather than renamed because
        it plays the same role -- the truncation with no assumption behind it -- and
        :attr:`CorrectionCheck.truncations_active` names it for what it is on each path.
    observation_clipped:
        How many ``(row, arm)`` cells of the **initial observation** mechanism
        :math:`\\hat\\pi(a, W)` lie outside ``[nuisance_bound, 1]``.  The counterpart of
        :attr:`initial_clipped` for the second mechanism a missing-outcome fit divides by,
        and ``0`` on a fit that has no such mechanism.

        It exists because the two are truncated **separately** and neither implies the
        other: a randomized trial's treatment mechanism is flat by design and cannot clip,
        while :math:`\\hat\\pi` is a fitted probability that can sit at its floor on a
        large share of rows.  Reading only :attr:`initial_clipped` there reports
        :attr:`CorrectionCheck.contract` as ``"theorem"`` on a fit whose observation
        mechanism is pinned, which is the one direction a scope label must not err in.
    observation_margin:
        The closest the targeted observation mechanism :math:`\\pi^*` comes to either end of
        ``[nuisance_bound, 1]``, as a fraction of that interval -- :attr:`margin`'s
        counterpart, two-sided for the same reason.  ``nan`` on a fit with no observation
        mechanism, which is a sentinel and not a zero: there is nothing to be close to.

        Two-sided rather than a floor alone because
        :func:`~cleverly.fluctuation.mechanism.solve_bounded_mechanism` is asked for a root
        in the whole interval, so a tilt driven to the upper cap is equally a solve the
        constraint changed -- and that, not "is the covariate large", is what
        :attr:`CorrectionCheck.contract` asks.
    solved:
        Whether this fit solved the equation the correction comes from. The flag is also the
        question of whether the term is in the reported curve, because
        :meth:`~cleverly.inference.influence.CorrectionParts.total` selects on the same
        guard.  ``False`` on a single-guard fit's other equation, and such a row is
        informational: :meth:`CorrectionCheck.correction_failures` does not read it, since
        a term nothing subtracts cannot make an interval wrong however large it is.

    :attr:`clip_bias` and :attr:`margin` are ``nan`` on the rows that do not carry them,
    which is a sentinel and not an unknown -- so two such rows are the same row, and
    :func:`~cleverly.utils.records.sentinel_equality` is what says so.  The generated
    ``__eq__`` said it only by accident and stopped saying it on Python 3.13; that module
    has the five-line reproduction.
    """

    draw: int
    arm: float
    label: str
    equation: str
    stored: float
    reported: float
    clipped: int
    solved: bool
    clip_bias: float = float("nan")
    margin: float = float("nan")
    initial_clipped: int = 0
    gr1_margin: float = float("nan")
    observation_clipped: int = 0
    observation_margin: float = float("nan")

    @property
    def residual(self) -> float:
        """:math:`\\Delta` -- the stored score minus the reported term's mean.

        ``nan`` when there is no stored score to compare against, which is not a pass: it
        is the absence of a check, and :attr:`solved` is what says so.
        """
        return self.stored - self.reported

    @property
    def name(self) -> str:
        """How this row is named wherever it is reported, arms by the caller's own label."""
        return f"mean ({self.equation})[{self.label}]"


@dataclass(frozen=True)
class CorrectionCheck:
    """Every arm's correction, against the score that was solved and against the tolerance.

    Empty -- and :attr:`passed` -- for any fit that reports no corrections, which is every
    fit but :class:`~cleverly.DRTMLE` with a non-empty guard.

    A guarded fit gets ``2 * arms`` rows whatever its guard is, of which ``len(guard)`` per
    arm are *judged*: the rest are the terms this fit's curve does not subtract, reported
    because they are the only thing that says what the guard left on the table.  Three
    things coincide on every row, and it is worth stating as one fact rather than three:
    the equation was solved, :attr:`CorrectionRow.stored` is finite, and the term is in the
    curve.
    """

    rows: tuple[CorrectionRow, ...]
    tolerance: float
    identity_tolerance: float
    n: int
    std_error: float
    #: The outcome scale the rows are reported on, so a reader can get back to the
    #: ``[0, 1]`` quantities the equations were solved in.
    scale: float = 1.0
    #: Name of the dataframe backend the fit's data arrived in, so that
    #: :meth:`to_frame` honours "results come back in the backend you passed in"
    #: without a caller having to thread the container back in by hand.
    backend: str | None = None

    @property
    def threshold(self) -> float:
        """The inferential bar a correction's mean is held to, ``tolerance * se / sqrt(n)``."""
        return float(self.tolerance * self.std_error / np.sqrt(self.n))

    @property
    def identity_threshold(self) -> float:
        """The bar an identity residual is held to, on the reported scale."""
        return float(self.identity_tolerance * max(1.0, self.scale))

    def identity_failures(self) -> tuple[CorrectionRow, ...]:
        """Rows where the solver's expression and the curve's are not the same one."""
        return tuple(
            row
            for row in self.rows
            if np.isfinite(row.residual) and abs(row.residual) > self.identity_threshold
        )

    def correction_failures(self) -> tuple[CorrectionRow, ...]:
        """Rows whose correction is not negligible at the state the fit returned.

        Only the rows the curve actually subtracts.  A term a fit did not solve for is not
        in its curve and cannot make its interval wrong however large it is, so an
        unsolved row is reported and judged against nothing -- see
        :attr:`CorrectionRow.solved`.  Dropping the ``row.solved`` here fails a correct
        single-guard fit, a mutation the partial-guard tests were watched
        against.
        """
        return tuple(row for row in self.rows if row.solved and abs(row.reported) > self.threshold)

    @property
    def passed(self) -> bool:
        return not self.identity_failures() and not self.correction_failures()

    def __bool__(self) -> bool:
        return self.passed

    @property
    def clipped(self) -> int:
        """The largest per-draw count of rows the truncation binds on at the exit.

        Zero on every converged bounded fit; :attr:`margin` says whether the bound had
        anything to do on this draw.
        """
        return max((row.clipped for row in self.rows), default=0)

    @property
    def margin(self) -> float:
        """The closest any draw's targeted mechanism comes to its bounds, as a fraction."""
        return min((row.margin for row in self.rows), default=float("nan"))

    @property
    def initial_clip_share(self) -> float:
        """The worst draw's share of ``(row, arm)`` pairs clipped at the *initial* mechanism.

        A share rather than the count :attr:`CorrectionRow.initial_clipped` carries, so that
        two sizes are comparable; the worst draw rather than the mean of them, because
        :attr:`contract` is a statement about the fit and one bound-active draw makes the
        whole report one.
        """
        arms = len({row.arm for row in self.rows})
        pairs = self.n * arms
        if not self.rows or pairs == 0:  # pragma: no cover - a guarded fit has both
            return float("nan")
        return max(row.initial_clipped for row in self.rows) / pairs

    @property
    def gr1_margin(self) -> float:
        """The closest any draw's :math:`g_{r,1}` comes to its bounds, as a fraction.

        Signed, so at or below zero says the truncation is active -- see
        :attr:`CorrectionRow.gr1_margin`.
        """
        return min((row.gr1_margin for row in self.rows), default=float("nan"))

    @property
    def has_observation_mechanism(self) -> bool:
        """Whether this fit targets an observation mechanism separately from ``g``.

        True exactly on a missing-outcome fit, which is the only construction here that
        divides by two mechanisms and truncates them at two different bounds.
        """
        return any(np.isfinite(row.observation_margin) for row in self.rows)

    @property
    def observation_clip_share(self) -> float:
        """The worst draw's share of cells clipped at the *initial* observation mechanism.

        ``nan`` -- not ``0.0`` -- on a fit with no such mechanism, because a zero would
        read as "nothing clipped" where the truth is "nothing to clip", and
        :attr:`truncations_active` must not name a truncation that does not exist.
        """
        arms = len({row.arm for row in self.rows})
        pairs = self.n * arms
        if not self.rows or pairs == 0 or not self.has_observation_mechanism:
            return float("nan")
        return max(row.observation_clipped for row in self.rows) / pairs

    @property
    def observation_margin(self) -> float:
        """The closest any draw's :math:`\\pi^*` comes to ``[nuisance_bound, 1]``."""
        return min((row.observation_margin for row in self.rows), default=float("nan"))

    @property
    def _truncation_count(self) -> int:
        """How many truncations this fit's contract is a statement about."""
        return 5 if self.has_observation_mechanism else 3

    @property
    def truncations_active(self) -> tuple[str, ...]:
        """Which of the relevant truncations bite on this fit, named.

        Empty on a fit inside the theorem-backed contract.  Named rather than counted
        because they are different objects with different standing: the mechanism ones are
        operations on something the theorem assumes bounded, and the last has no assumption
        in the theorem at all.

        **Three on a complete-data fit and five on a missing-outcome one.**  That fit
        divides by two mechanisms, truncated separately at two different bounds, and
        neither implies the other -- a randomized trial's treatment mechanism is flat by
        design while its observation mechanism is fitted and can sit at its floor.  The two
        extra names are ``nan``-guarded rather than zero-guarded, so a complete-data fit
        reports exactly the three it always did.
        """
        active = []
        if np.isfinite(self.initial_clip_share) and self.initial_clip_share > 0.0:
            active.append("g-hat at the initial fit")
        if np.isfinite(self.observation_clip_share) and self.observation_clip_share > 0.0:
            active.append("pi-hat at the initial fit")
        if np.isfinite(self.margin) and self.margin <= MARGIN_ACTIVE:
            active.append("g* at the exit")
        if np.isfinite(self.observation_margin) and self.observation_margin <= MARGIN_ACTIVE:
            active.append("pi* at the exit")
        if np.isfinite(self.gr1_margin) and self.gr1_margin <= MARGIN_ACTIVE:
            # `g_r1` does not exist in the missing-outcome construction; the column carries
            # the gamma reductions there, so the label says which object it is about.
            active.append("gamma_a / gamma_m" if self.has_observation_mechanism else "g_r1")
        return tuple(active)

    @property
    def contract(self) -> str:
        """Which estimator this fit's numbers are evidence about.

        ``"theorem"`` where none of the relevant truncations is active, in which case
        :func:`~cleverly.fluctuation.mechanism.solve_bounded_mechanism` returned the
        unconstrained solve bit for bit and the fit **is** Theorem 1's estimator;
        ``"bound-active"`` otherwise, which is *empirically supported and outside the
        theorem* rather than wrong -- see this module's docstring.  ``"none"`` for a fit that
        reports no corrections and so has no mechanism tilt to ask about.

        **Not a verdict.**  :attr:`passed` does not read this, and a bound-active fit whose
        identities hold and whose scores are negligible has passed every check here.
        """
        if not self.rows:
            return "none"
        return "bound-active" if self.truncations_active else "theorem"

    def to_frame(self, data: Any = None) -> Any:
        payload = {
            "draw": [row.draw for row in self.rows],
            "arm": [row.label for row in self.rows],
            "equation": [row.equation for row in self.rows],
            "stored": [row.stored for row in self.rows],
            "reported": [row.reported for row in self.rows],
            "residual": [row.residual for row in self.rows],
            "clip_bias": [row.clip_bias for row in self.rows],
            "clipped": [row.clipped for row in self.rows],
            "margin": [row.margin for row in self.rows],
            "initial_clipped": [row.initial_clipped for row in self.rows],
            "gr1_margin": [row.gr1_margin for row in self.rows],
            "observation_clipped": [row.observation_clipped for row in self.rows],
            "observation_margin": [row.observation_margin for row in self.rows],
            # Which rows `reported` is being judged on. Without it a partial-guard frame
            # carries a large number in a column of small ones and says nothing about why.
            "solved": [row.solved for row in self.rows],
        }
        return emit_frame(payload, data, backend=self.backend)

    def summary(self) -> str:
        if not self.rows:
            return "Correction check\n" + "-" * 16 + "\nThis fit reports no corrections."
        return "\n".join(
            [
                "Correction check",
                "-" * 16,
                format_table(
                    ["draw", "arm", "equation", "solved score", "reported", "residual", "B_clip"],
                    [
                        [
                            str(row.draw),
                            row.label,
                            row.equation,
                            "-" if not np.isfinite(row.stored) else f"{row.stored:.3e}",
                            f"{row.reported:.3e}",
                            "-" if not np.isfinite(row.residual) else f"{row.residual:.3e}",
                            "-" if not np.isfinite(row.clip_bias) else f"{row.clip_bias:.3e}",
                        ]
                        for row in self.rows
                    ],
                ),
                "",
                *self._notes(),
                self._verdict(),
            ]
        )

    def _notes(self) -> list[str]:
        notes = [
            f"  scores on the outcome scale (x{self.scale:.4g} from the fitting scale); "
            f"tolerance {self.threshold:.2e}, identity {self.identity_threshold:.2e}",
            f"  the truncation binds on up to {self.clipped} row(s) of {self.n} at the exit "
            f"of any one draw, and the targeted treatment mechanism comes within "
            f"{self.margin:.2g} of a bound"
            + (
                f", the targeted observation mechanism within {self.observation_margin:.2g}"
                if self.has_observation_mechanism
                else ""
            )
            + "; the per-draw figures are in to_frame()",
            # Item 25's scope label, on the face of the fit rather than recomputable from it.
            # Deliberately not in `_verdict`: which estimator a number is evidence about is a
            # different question from whether the fit solved what it reports, and reading it
            # as a verdict would report a sound bound-active fit as broken.
            f"  contract: {self.contract}"
            + (
                f" -- {', '.join(self.truncations_active)} active, so this fit is empirically "
                "supported and outside Theorem 1 rather than "
                "wrong; it is not a failure and the verdict below does not read it"
                if self.truncations_active
                else f" -- none of the {self._truncation_count} truncations is active, so "
                "this fit is Theorem 1's estimator: clip share 0 at the initial "
                f"mechanism{'s' if self.has_observation_mechanism else ''}, and margins "
                + (
                    f"{self.margin:.2g} (g*), {self.observation_margin:.2g} (pi*) and "
                    f"{self.gr1_margin:.2g} (gamma_a/gamma_m)"
                    if self.has_observation_mechanism
                    else f"{self.margin:.2g} (g*) and {self.gr1_margin:.2g} (g_r1)"
                )
            ),
        ]
        unsolved = {row.equation for row in self.rows if not row.solved}
        if unsolved:
            notes.append(
                f"  {', '.join(sorted(unsolved))} is not among the equations this fit's guard= "
                "solves, so the reported curve does not subtract it; the mean above is a "
                "diagnostic and is held to nothing"
            )
        return [*notes, ""]

    def _verdict(self) -> str:
        identity, correction = self.identity_failures(), self.correction_failures()
        if not identity and not correction:
            return (
                "PASS: every correction is a negligible mean of the same expression the "
                "targeting step solved."
            )
        lines = []
        if identity:
            lines.append(
                f"FAIL: {len(identity)} state identit{'y' if len(identity) == 1 else 'ies'} "
                "not satisfied -- the score the targeting step recorded and the term the "
                "reported curve carries are not the same functional of the returned state. "
                "That is a software defect, not a fit that failed "
                "to converge, and the standard errors do not describe this estimate."
            )
        if correction:
            lines.append(
                f"FAIL: {len(correction)} correction(s) above {self.threshold:.2e} at the "
                "state this fit returned, so the reported curve is not centred."
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def correction_check(
    result: TMLEResult,
    *,
    tolerance: float,
    identity_tolerance: float = IDENTITY_TOLERANCE,
    std_error: float | None = None,
) -> CorrectionCheck:
    """Recompute every arm's corrections from the state each draw returned.

    Free: array arithmetic on what the fit already carries, refitting nothing.  Every draw,
    never draw zero alone -- a repeated fit averages curves whose defects do not average
    away, and the draw that clips is not usually the first.
    """
    data = result.data
    weights = np.asarray(data.weights, dtype=float).reshape(-1)
    reference = _reference_se(result) if std_error is None else float(std_error)

    rows: list[CorrectionRow] = []
    for draw, repeat in enumerate(result.repeats):
        scaler = repeat.nuisance.scaler
        scaled = scaler.scale(data.outcome)
        # The report is on the outcome's own scale, which is the scale the estimate and its
        # standard error are on. Every quantity below is a linear functional of the scaled
        # residual, so one factor converts all of them -- and leaving it out would compare
        # a correction against a threshold `range` times too tight (or too loose), which is
        # lesson 8's pattern in the place it is easiest to miss.
        scale = float(scaler.range)
        for fluctuation in repeat.fluctuations.values():
            reduction = fluctuation.reduction
            if reduction is None:
                continue
            parts = correction_parts(
                data, repeat.nuisance, fluctuation, fluctuation.targeted, scaled
            )
            if parts is None:  # pragma: no cover - reduction implies parts
                continue
            clipped = int(np.count_nonzero(parts.clipped))
            mechanism = fluctuation.mechanism
            margin = _margin(
                reported_mechanism(repeat.nuisance, fluctuation, reduction.reduced.arms),
                reduction.bounds,
            )
            if parts.d_a is not None and parts.d_m is not None and parts.d_y is not None:
                observation = reduction.observation
                missingness = repeat.nuisance.missingness
                if (
                    mechanism is None
                    or observation is None
                    or missingness is None
                    or reduction.missingness_bound is None
                ):
                    raise ValueError(
                        "a missing-outcome correction needs a targeted treatment mechanism, "
                        "a targeted observation mechanism, the initial observation "
                        "mechanism, and the bound both of the latter were formed at"
                    )
                missingness_bound = float(reduction.missingness_bound)
                observation_bounds = (missingness_bound, 1.0)
                initial = np.asarray(repeat.nuisance.propensity.values, dtype=float)
                lower, upper = reduction.bounds
                initial_clipped = int(np.count_nonzero((initial < lower) | (initial > upper)))
                # The second mechanism this fit divides by, at the second bound.  It is
                # counted separately from `initial_clipped` and not folded into it because
                # the two are truncated separately and neither implies the other: on a
                # randomized trial `g-hat` is flat by design and cannot clip, so a fit
                # reading only the propensity reports `contract == "theorem"` however much
                # of `pi-hat` is pinned at its floor.
                initial_pi = np.asarray(missingness, dtype=float)
                observation_clipped = int(
                    np.count_nonzero((initial_pi < missingness_bound) | (initial_pi > 1.0))
                )
                observation_margin = _margin(observation.propensity, observation_bounds)
                gamma_margin = min(
                    _margin(reduction.reduced.gamma_a, reduction.bounds),
                    _margin(reduction.reduced.gamma_m, observation_bounds),
                )
                stored = {
                    "D*_A": np.asarray(mechanism.score),
                    "D*_M": np.asarray(observation.score),
                    "D*_Y": np.asarray(reduction.score),
                }
                reported = {"D*_A": parts.d_a, "D*_M": parts.d_m, "D*_Y": parts.d_y}
                for column, arm in enumerate(reduction.reduced.arms):
                    for equation in ("D*_A", "D*_M", "D*_Y"):
                        values = stored[equation]
                        rows.append(
                            CorrectionRow(
                                draw=draw,
                                arm=float(arm),
                                label=str(data.arm_label(arm)),
                                equation=equation,
                                stored=scale * float(values[column]),
                                reported=scale * float(np.mean(weights * reported[equation][arm])),
                                clipped=clipped,
                                margin=margin,
                                initial_clipped=initial_clipped,
                                gr1_margin=gamma_margin,
                                observation_clipped=observation_clipped,
                                observation_margin=observation_margin,
                                solved=True,
                            )
                        )
                continue
            # Item 25's other two witnesses. Both are properties of this draw rather than of
            # a single arm's equation, so they ride on every row of it exactly as `clipped`
            # and `margin` do -- and both come off the arrays the covariates divide by: the
            # untruncated initial mechanism `Propensity.bounded` was applied to, and the
            # untruncated `gr1` `bounded_gr1` was applied to. This branch is complete-data
            # only -- a missing-outcome fit returns above -- so the propensity is the whole
            # of what equation (8)'s covariate divides by and reading it here is exact. The
            # second mechanism, and the second bound, exist only on the branch that has
            # `observation_clipped` and `observation_margin` to report them.
            initial_fit = repeat.nuisance.propensity
            initial = np.column_stack([initial_fit.arm(arm) for arm in reduction.reduced.arms])
            lower, upper = reduction.bounds
            initial_clipped = int(np.count_nonzero((initial < lower) | (initial > upper)))
            gr1_margin = _margin(reduction.reduced.gr1, reduction.bounds)
            stored_g = np.asarray(mechanism.score) if mechanism is not None else np.zeros(0)
            stored_q = np.asarray(reduction.score)
            for column, arm in enumerate(reduction.reduced.arms):
                # `reduced_mechanism_covariate` and `reduced_outcome_submodel` both build
                # their columns in `reduced.arms` order, so the stored score's column and
                # this arm's correction are the same equation -- read by position rather
                # than by a name neither array carries.
                rows.append(
                    CorrectionRow(
                        draw=draw,
                        arm=float(arm),
                        label=str(data.arm_label(arm)),
                        equation="D*_g",
                        stored=(
                            scale * float(stored_g[column])
                            if stored_g.size > column
                            else float("nan")
                        ),
                        reported=scale * float(np.mean(weights * parts.d_g[arm])),
                        clip_bias=scale * float(np.mean(weights * parts.clip_bias[arm])),
                        clipped=clipped,
                        margin=margin,
                        initial_clipped=initial_clipped,
                        gr1_margin=gr1_margin,
                        # `parts.guard` rather than `reduction.guard`: the selection has to
                        # come from the object the curve selected with, for the same reason
                        # the means come from the arrays the curve carries.
                        solved="Q" in parts.guard,
                    )
                )
                rows.append(
                    CorrectionRow(
                        draw=draw,
                        arm=float(arm),
                        label=str(data.arm_label(arm)),
                        equation="D*_Q",
                        stored=(
                            scale * float(stored_q[column])
                            if stored_q.size > column
                            else float("nan")
                        ),
                        reported=scale * float(np.mean(weights * parts.d_q[arm])),
                        clipped=clipped,
                        margin=margin,
                        initial_clipped=initial_clipped,
                        gr1_margin=gr1_margin,
                        solved="g" in parts.guard,
                    )
                )
    return CorrectionCheck(
        rows=tuple(rows),
        tolerance=tolerance,
        identity_tolerance=identity_tolerance,
        n=int(data.n),
        std_error=reference,
        scale=float(result.nuisance.scaler.range),
        backend=result.data.backend,
    )


def _margin(mechanism: Any, bounds: tuple[float, float]) -> float:
    """How close ``mechanism`` comes to either bound, as a fraction of the interval.

    Small means the constrained root is pressed against the feasible set's boundary, which
    is what "the truncation binds" looks like once the solve is constrained: no row is
    outside the bounds -- that is the point of the constraint -- so a count of rows outside
    them says nothing, and this says what the count used to.  Zero when a row sits exactly on
    a bound, which the bounded solve permits and the unconstrained one never produced.
    """
    lower, upper = float(bounds[0]), float(bounds[1])
    values = np.asarray(mechanism, dtype=float).reshape(-1)
    if values.size == 0 or not upper > lower:  # pragma: no cover - a fit always has both
        return float("nan")
    return float(np.min(np.minimum(values - lower, upper - values)) / (upper - lower))


def _reference_se(result: TMLEResult) -> float:
    """The largest reported standard error, which is what the score tolerance is built on."""
    return max(
        (
            float(estimate.std_error)
            for estimate in result.estimates.values()
            if np.isfinite(estimate.std_error)
        ),
        default=1.0,
    )
