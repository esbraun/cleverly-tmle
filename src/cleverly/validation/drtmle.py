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
reported their scores solved to ``1e-11``.  That was ``docs/roadmap.md``'s item 20, it
accounted for item 11, and this module is the instrument that made it impossible to hide: it
recomputes each correction's empirical mean **from the exact returned state**, compares it
with the score the solver recorded, and reports the discrepancy as :math:`\Delta_g` and
:math:`\Delta_Q` alongside the clipping bias :math:`B_{clip}` that explained it.

**Both items are closed and this module is why they could be.**  Piece B1b replaced the
solver at the ``DRTMLE`` call sites with
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
it.  This is where item 23 was found and it is the wrong way round from how it read then.
The instrument's first run against a ``guard=("g",)`` fit reported :math:`2.8\times10^{-3}`
at arm 1 with **no** row clipped and no equation (9) anywhere in the fit -- and at the time
the curve subtracted that term anyway.  It no longer does; the row that found it stays.

What this does **not** do is choose a convention, and it did not need to: every identity here
is valid under all four candidates, which is what let it land a piece ahead of the one that
chose.  Which mechanism equation (9) ought to be solved against was a derivation rather than
a taste -- the theorem's own algorithm truncates nothing at all, so there was no convention
in the source to match and no document to wait for -- and it is piece B1b, whose reasoning is
on :func:`~cleverly.fluctuation.mechanism.solve_bounded_mechanism`.

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
  witness is :attr:`CorrectionRow.margin` -- **not** :attr:`CorrectionRow.clipped`, which B1b
  emptied.  The alternation carries the truncated array forward, so a converged fit clips
  nothing at the exit however hard the draw was, and a fixture selected on the exit count
  would now be selected from nothing at all: ``docs/roadmap.md``'s stop-ship 14, a check
  agreeing where it could not have disagreed, arriving in a second place.  What a constrained
  root does instead is sit *against* the boundary, and the margin is how that shows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..estimators.base import format_table
from ..estimators.tmle import correction_parts

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult

__all__ = ["IDENTITY_TOLERANCE", "CorrectionCheck", "CorrectionRow", "correction_check"]

#: How far a state identity may miss before it is called a defect, on the outcome's scale.
#: A number rather than a judgement, and the gap either side of it is what makes it one:
#: measured on ``nonlinear_dgp`` at ``n = 400``, the residual is ``2e-19`` on a draw where
#: the identity holds and ``1.3e-04`` on one where it does not, with the smallest observed
#: real failure at ``7e-08``.  So this sits seven orders above the arithmetic and four
#: below the defect, and it is deliberately *not* expressed relative to the score: the
#: quantity is a difference of two evaluations of one expression, and its right value is
#: zero rather than something small compared with anything.
IDENTITY_TOLERANCE = 1e-12


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
        for the other.  Before B1b it reproduced **minus** :attr:`residual` to floating
        point, which is what made it a check on item 20's diagnosis rather than merely a new
        column; the sign is the orientation ``docs/drtmle-validation-plan.md`` defines it
        in, and the two differed because one residual read :math:`1_a - g` and the other
        :math:`1_a - g^b`.  It is **zero now**, on every fit, because there is no longer a
        raw tilted mechanism for it to measure the distance to.
    clipped:
        How many rows the truncation binds on *at the exit*.  Zero on every fit since B1b,
        and that is the point rather than a defect in the column: the alternation carries
        the truncated array forward, so at a fixed point :math:`\\epsilon \\to 0` and the raw
        and truncated tilts coincide.  What it says now is that the fix took.  Use
        :attr:`margin` to ask whether the draw was a hard one.
    margin:
        The closest the targeted mechanism comes to either bound, as a fraction of the
        interval between them: ``min_i min(g*_i - lo, hi - g*_i) / (hi - lo)``.  This is the
        witness that replaces :attr:`clipped`, and it says what that column used to say --
        **whether the truncation had anything to do on this draw** -- in the only form that
        survives B1b.  A constrained root sits *at* the boundary of the feasible set, so a
        draw whose unconstrained tilt wanted to leave the bounds comes back with the
        mechanism pressed against one: measured at ``1.2e-06`` on the draw item 20 was found
        on, against ``0.14`` on its sibling that never clipped.  Five orders, and no
        threshold inside the column -- a test picks its own.

        It is not a proof that the constraint was active, and nothing derivable from the
        returned arrays is: the trajectory that got there is not on the record.  What it is
        is a property that separates the two draws by five orders and cannot be manufactured
        by the fix, which is what a fixture's precondition has to be.
    solved:
        Whether this fit solved the equation the correction comes from -- which, since
        item 23, is the same question as whether the term is in the reported curve, because
        :meth:`~cleverly.inference.influence.CorrectionParts.total` selects on the same
        guard.  ``False`` on a single-guard fit's other equation, and such a row is
        informational: :meth:`CorrectionCheck.correction_failures` does not read it, since
        a term nothing subtracts cannot make an interval wrong however large it is.
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
        single-guard fit, which is one of the mutations item 23's tests were watched
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

        Zero on every fit since B1b; :attr:`margin` is what says whether the bound had
        anything to do on this draw.
        """
        return max((row.clipped for row in self.rows), default=0)

    @property
    def margin(self) -> float:
        """The closest any draw's targeted mechanism comes to its bounds, as a fraction."""
        return min((row.margin for row in self.rows), default=float("nan"))

    def to_frame(self, data: Any = None) -> Any:
        from ..utils.frames import frame_from_dict

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
            # Which rows `reported` is being judged on. Without it a partial-guard frame
            # carries a large number in a column of small ones and says nothing about why.
            "solved": [row.solved for row in self.rows],
        }
        if data is not None:
            return data.frame_like(payload)
        return frame_from_dict(payload)

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
            f"of any one draw, and the targeted mechanism comes within {self.margin:.2g} of "
            "a bound; the per-draw figures are in to_frame()",
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
                "That is a software defect (docs/roadmap.md item 20), not a fit that failed "
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
                mechanism.propensity
                if mechanism is not None
                else repeat.nuisance.propensity.arm(reduction.reduced.arms[1]),
                reduction.bounds,
            )
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
