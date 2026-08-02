r"""Did the targeting step actually work?

TMLE's guarantees are built on the estimator solving the efficient influence-function
equation,

.. math:: P_n \hat D^*(O) = \frac{1}{n}\sum_i \hat D^*_i = 0.

If that equation is not solved, the estimate is not the TMLE -- it is a plug-in
estimate with an unquantified bias, and its influence-curve standard error does not
describe it.  Unlike the identification assumptions this is a *checkable* condition,
and there is no reason not to check it on every fit.

What a pass does and does not mean
----------------------------------

Read it as necessary, not sufficient.  What is checked is that the fluctuation reached
the root of the equation the library posed -- with :math:`\hat D^*` being whatever
:mod:`cleverly.inference.influence` computes.  A wrong clever covariate used
*consistently*, in both the targeting step and the influence curve, would solve its own
equation to machine precision and pass here.  So this is an implementation invariant of
the fit in front of you, not evidence that the estimating equation behind it is the
right one.

That the equation is the right one is established elsewhere, and separately:

* ``tests/unit/test_influence_gateaux.py`` checks :math:`\hat D^*` against the numerical
  Gateaux derivative of the target parameter on a finite-support law -- the definition of
  the efficient influence function, computed without any of the library's machinery;
* ``tests/unit/test_remainder.py`` checks that the resulting estimating equation has the
  second-order product remainder that double robustness consists of;
* ``tests/e2e/test_oracle.py`` cross-checks the point estimate against an independently
  written AIPW estimator, and ``tests/e2e/test_double_robustness.py`` runs the full
  misspecification grid end to end.

A *failure* here, by contrast, is informative on its own and non-circular: it means the
solver did not converge, or the fluctuation could not reach the root.

On a doubly-robust fit it is not the efficient equation
---------------------------------------------------------

Everything above is written for :math:`\hat D^*`, and a :class:`~cleverly.DRTMLE` fit
solves three equations rather than one.  The extra two are not refinements of the
efficient score equation: the curve their solution leaves is
:math:`D = D^* - D^*_Q - D^*_g`, the **estimator's** asymptotic influence function at the
nuisance limits, and it is generally not the canonical gradient at :math:`P_0`.  When both
nuisances are consistent the corrections vanish row by row and the two coincide -- which is
precisely the case the variant is not for.  So the verdict here branches on
:attr:`ScoreCheck.corrected` and does not sign such a fit off as having solved "the
estimated efficient score equation"; see
:func:`~cleverly.inference.influence.reduced_corrections` and
:mod:`cleverly.estimators.drtmle`.

The check compares :math:`|P_n \hat D^*|` against the standard error.  The natural scale is
:math:`\widehat{se} / \sqrt{n}`: the score has to be small relative to the estimate's
own sampling variability, not merely small in absolute terms, since the units of the
outcome are arbitrary.  A converged fit typically lands near machine precision, many
orders of magnitude below the tolerance.

The usual reasons for a failure:

* **positivity.** Targeted predictions pin against ``[0, 1]`` and cannot move further,
  so the fluctuation cannot reach the root.  Check
  :meth:`~cleverly.sensitivity.SensitivityAnalysis.positivity`.
* **a fluctuation that has to travel far**, from a poor initial fit -- try
  ``targeting="one_step"``, which rebuilds the direction along the path.
* **too coarse a step** in the one-step walk -- lower ``step_size``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..estimators.base import format_table
from .drtmle import CorrectionCheck, correction_check

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult

__all__ = ["ScoreCheck", "ScoreCheckRow", "score_check"]

#: The score must be below ``tolerance * se / sqrt(n)`` to pass.  A converged
#: fluctuation is normally near machine precision, so this is a loose gate that only
#: catches genuine failures.
DEFAULT_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ScoreCheckRow:
    """The score diagnostic for one estimand or one targeted family."""

    name: str
    kind: str
    score: float
    threshold: float
    std_error: float
    passed: bool
    converged: bool
    n_iter: int
    method: str
    #: The same score before the fluctuation moved anything.  A step that starts near
    #: zero had nothing to do, which is a different situation from one that started
    #: large and was driven down -- and only the latter is evidence targeting worked.
    score_initial: float = float("nan")
    #: Why the step stopped, when it did not converge.  Empty when it did.
    failure: str = ""
    #: Conditioning of the last Newton Hessian.  Large means ``epsilon`` is barely
    #: identified even where the score looks solved.
    hessian_condition: float = float("nan")
    #: For a fold-targeted fit, how many of the per-fold solves converged.
    folds_converged: tuple[int, int] | None = None

    @property
    def reduction(self) -> float:
        """How far targeting moved the score, as a factor.  Above 1 means it shrank."""
        if not np.isfinite(self.score_initial) or self.score <= 0:
            return float("nan")
        return abs(self.score_initial) / abs(self.score)

    @property
    def ratio(self) -> float:
        """Score relative to its threshold; below 1 passes."""
        if self.threshold <= 0:
            return float("nan")
        return abs(self.score) / self.threshold


@dataclass(frozen=True)
class ScoreCheck:
    """Whether the targeting step solved the score equations it was supposed to.

    "The efficient score equation" for a plain fit, and deliberately not that phrase for a
    doubly-robust one -- see :attr:`corrected`.
    """

    rows: tuple[ScoreCheckRow, ...]
    tolerance: float
    n: int
    #: Whether this fit reports the **corrected** curve
    #: :math:`D = D^* - D^*_Q - D^*_g` rather than :math:`D^*`.  True for a
    #: :class:`~cleverly.DRTMLE` fit with a non-empty guard, and what the verdict branches
    #: on: the equations such a fit solves are its own three, and the curve they leave is
    #: the *estimator's* influence function at the nuisance limits, which is generally not
    #: the efficient one.  Signing that off as "the estimated efficient score equation"
    #: was the one place the package asserted the thing
    #: :func:`~cleverly.inference.influence.reduced_corrections` exists to deny.
    corrected: bool = False
    #: The per-arm corrections behind this check's ``identity`` and ``correction`` rows,
    #: for a reader who wants the recomputation itself rather than the verdict on it.
    #: ``None`` on a check built by hand; empty rows on any fit that reports no
    #: corrections, which is every fit but a guarded :class:`~cleverly.DRTMLE` one.
    corrections: CorrectionCheck | None = None

    @property
    def passed(self) -> bool:
        """True when every row is within tolerance."""
        return all(row.passed for row in self.rows)

    @property
    def identity_failures(self) -> tuple[ScoreCheckRow, ...]:
        """Failures that are software defects rather than unsolved equations.

        A row here says the score the targeting step recorded and the term the reported
        curve carries are not the same functional of the state the fit returned --
        ``docs/roadmap.md``'s item 20.  Iterating longer cannot fix one, which is why it is
        worded apart from every other failure this class reports.
        """
        return tuple(row for row in self.failures if row.kind == "identity")

    @property
    def passed_apart_from_identities(self) -> bool:
        """Whether every failure here is a state identity rather than an unsolved equation.

        Not a softer :attr:`passed` -- a fit in this state is still not one to report from.
        It is what lets a verdict name one cause where there is one cause.
        """
        return all(row.kind == "identity" for row in self.failures)

    def __bool__(self) -> bool:
        return self.passed

    @property
    def failures(self) -> tuple[ScoreCheckRow, ...]:
        return tuple(row for row in self.rows if not row.passed)

    def to_frame(self, data: Any = None) -> Any:
        from ..utils.frames import frame_from_dict

        payload = {
            "name": [row.name for row in self.rows],
            "kind": [row.kind for row in self.rows],
            "score": [row.score for row in self.rows],
            "threshold": [row.threshold for row in self.rows],
            "ratio": [row.ratio for row in self.rows],
            "passed": [row.passed for row in self.rows],
            "n_iter": [row.n_iter for row in self.rows],
            "method": [row.method for row in self.rows],
        }
        if data is not None:
            return data.frame_like(payload)
        return frame_from_dict(payload)

    def one_line(self) -> str:
        """The verdict as a block a report can append, without the table.

        Phrased here rather than at the call site so that every place a verdict is spoken
        -- this module's own report, and :meth:`cleverly.TMLEResult.summary` -- says the
        same thing.  It names the failing rows, because "the check failed" without them
        sends the reader back to the method call this line exists to spare them.
        """
        failures = self.failures
        if not failures:
            return (
                f"score check: PASS -- all {len(self.rows)} within tolerance "
                f"(worst |score| {self._worst_ratio():.2e} of its threshold)."
            )
        named = "; ".join(
            f"{row.name} |score| {abs(row.score):.3e} against {row.threshold:.3e}"
            for row in failures
        )
        return "\n".join(
            [
                f"score check: FAIL -- {len(failures)} of {len(self.rows)} not solved.",
                f"  {named}",
                *self._identity_lines(),
                "  The standard errors above are read off an influence curve whose mean is",
                "  not zero, so they do not describe this estimate.  See",
                "  res.validation.score_check() for the table and cleverly.validation.score",
                "  for the usual causes.",
            ]
        )

    def _identity_lines(self) -> list[str]:
        """The state-identity failures, said as what they are and not as a convergence one.

        An unsolved equation is a fit that did not get there; a broken identity is a fit
        that solved something else.  Reporting the second in the first's words -- "try
        one_step, lower the step size" -- would send a reader looking for a numerical
        problem that is not there, which is precisely what happened for two revisions while
        the loop reported ``1e-11`` and the curve was out by ``2e-04``.
        """
        failures = self.identity_failures
        if not failures:
            return []
        return [
            f"  {len(failures)} of those {'is' if len(failures) == 1 else 'are'} a state "
            "identity, not an unsolved equation: the score the targeting",
            "  step recorded and the term the reported curve carries are not the same",
            "  functional of the state this fit returned.  That is a defect in the",
            "  implementation (docs/roadmap.md item 20) and iterating longer will not fix",
            "  it.  See res.validation.correction_check().",
        ]

    def _worst_ratio(self) -> float:
        ratios = [row.ratio for row in self.rows if np.isfinite(row.ratio)]
        return max(ratios) if ratios else float("nan")

    def summary(self) -> str:
        if not self.passed:
            verdict = (
                "FAIL: the score equation was not solved -- the influence-curve standard "
                "errors this fit reports do not describe this estimate. See the module "
                "docstring for the usual causes."
            )
            if self.identity_failures:
                # Named as its own thing and *first*, because the other failing rows on
                # such a fit are usually this one's consequence: a curve built from an
                # expression the loop did not solve is not centred, so the correction rows
                # and the per-estimand rows go with it. Which rows failed is in the table
                # above; what a reader cannot get from the table is that iterating longer
                # is not the remedy.
                verdict = "\n".join(
                    [
                        "FAIL: a state identity does not hold -- the score the targeting "
                        "step recorded and the",
                        "term the reported curve carries are not the same functional of "
                        "the state this fit",
                        "returned. That is a defect in the implementation "
                        "(docs/roadmap.md item 20) rather than",
                        "a fit that failed to converge, and the standard errors do not "
                        "describe this estimate.",
                        "See res.validation.correction_check() for the recomputation and "
                        "the clipping bias.",
                    ]
                )
        elif self.corrected:
            # Not "the efficient score equation": this fit solved *its own* equations, and
            # the curve they leave is the estimator's influence function at the nuisance
            # limits. The two coincide when both nuisances are consistent, because the
            # corrections then vanish row by row -- and that is exactly the case the
            # variant is not for.
            solved = sum(1 for row in self.rows if row.kind == "fluctuation")
            verdict = "\n".join(
                [
                    f"PASS: the targeting step solved all {solved} estimated score equations "
                    "of the doubly-robust estimator.",
                    "Validity is not efficiency: the curve reported is D = D* - D*_Q - D*_g, "
                    "entitled to be believed",
                    "under weaker conditions than D* rather than efficient under them. See "
                    "cleverly.estimators.drtmle.",
                ]
            )
        else:
            verdict = "PASS: the targeting step solved the estimated efficient score equation."
        return "\n".join(
            [
                "Score-equation check",
                "-" * 20,
                format_table(
                    ["target", "kind", "|score|", "before", "threshold", "ratio", "ok"],
                    [
                        [
                            row.name,
                            row.kind,
                            f"{abs(row.score):.3e}",
                            f"{abs(row.score_initial):.3e}"
                            if np.isfinite(row.score_initial)
                            else "-",
                            f"{row.threshold:.3e}",
                            f"{row.ratio:.2e}",
                            "yes" if row.passed else "NO",
                        ]
                        for row in self.rows
                    ],
                ),
                *self._notes(),
                "",
                verdict,
            ]
        )

    def _notes(self) -> list[str]:
        """Anything the table has no column for: named failures, fold convergence."""
        notes: list[str] = []
        for row in self.rows:
            if row.failure:
                # A recomputed row's `failure` is a statement about the fit's arithmetic
                # rather than about a solver that gave up, so it is not introduced as one.
                stopped = "" if row.kind in ("correction", "identity") else "targeting stopped -- "
                notes.append(f"  {row.name}: {stopped}{row.failure}")
            if row.folds_converged is not None:
                good, total = row.folds_converged
                if good < total:
                    notes.append(f"  {row.name}: {total - good} of {total} folds did not converge")
            if np.isfinite(row.hessian_condition) and row.hessian_condition > 1e8:
                notes.append(
                    f"  {row.name}: Hessian condition {row.hessian_condition:.2e} -- epsilon is "
                    "barely identified, so a solved score may still be fragile"
                )
        return ["", *notes] if notes else []

    def raise_if_failed(self) -> None:
        """Raise when the check failed -- for use in pipelines that must not ship a bad fit."""
        if self.passed:
            return
        names = ", ".join(row.name for row in self.failures)
        raise AssertionError(
            f"the TMLE score equation was not solved for: {names}. {self.summary()}"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def score_check(result: TMLEResult, *, tolerance: float = DEFAULT_TOLERANCE) -> ScoreCheck:
    """Check that targeting solved the score equation.

    Two things are verified, and both matter:

    * per targeted family, the fluctuation's own score
      ``mean(w * h * (Y - Q*))`` -- this is the equation the fluctuation is
      *solving*;
    * per estimand, the mean of its influence curve -- this is the equation the
      *estimate* relies on, which for a derived estimand such as the risk ratio is a
      transformation of the first.
    """
    n = result.data.n
    rows: list[ScoreCheckRow] = []

    reference_se = max(
        (
            estimate.std_error
            for estimate in result.estimates.values()
            if np.isfinite(estimate.std_error)
        ),
        default=1.0,
    )

    # Every draw's fluctuation, not just the first. Each solved its own score equation,
    # and a draw whose Newton step failed contributes to the reported estimate exactly as
    # the others do -- checking one of R would let that failure through silently. The row
    # name carries the draw index only when there is more than one, so an ordinary fit's
    # report is unchanged.
    for index, repeat in enumerate(result.repeats):
        for group, fluctuation in repeat.fluctuations.items():
            threshold = tolerance * reference_se / np.sqrt(n)
            score = fluctuation.score_norm
            rows.append(
                ScoreCheckRow(
                    name=group if result.n_repeats == 1 else f"{group}[draw {index}]",
                    kind="fluctuation",
                    score=score,
                    threshold=float(threshold),
                    std_error=float(reference_se),
                    passed=bool(score <= threshold),
                    converged=fluctuation.converged,
                    n_iter=fluctuation.n_iter,
                    method=fluctuation.method,
                    score_initial=fluctuation.initial_score_norm,
                    failure=fluctuation.failure or "",
                    hessian_condition=fluctuation.hessian_condition,
                    folds_converged=(
                        (sum(f.converged for f in fluctuation.folds), len(fluctuation.folds))
                        if fluctuation.folds
                        else None
                    ),
                )
            )
            # A group whose parameter is defined through the mechanism solves *two*
            # equations, and the per-estimand rows below check only their sum -- the
            # influence curve contains both terms, so its mean cannot be zero unless
            # both are solved, but a failure there does not say which. This row does.
            if fluctuation.mechanism is not None:
                mechanism = fluctuation.mechanism
                score = float(np.max(np.abs(mechanism.score))) if mechanism.score.size else 0.0
                stem = f"{group} (mechanism)"
                rows.append(
                    ScoreCheckRow(
                        name=stem if result.n_repeats == 1 else f"{stem}[draw {index}]",
                        kind="fluctuation",
                        score=score,
                        threshold=float(threshold),
                        std_error=float(reference_se),
                        passed=bool(score <= threshold),
                        converged=mechanism.converged,
                        n_iter=len(mechanism.trace),
                        method=fluctuation.method,
                        score_initial=(
                            float(np.max(np.abs(mechanism.score_initial)))
                            if mechanism.score_initial.size
                            else float("nan")
                        ),
                        failure=mechanism.failure or "",
                        hessian_condition=(
                            float("nan")
                            if mechanism.hessian_condition is None
                            else mechanism.hessian_condition
                        ),
                    )
                )
            # A doubly-robust fit solves a *third* equation, in a second submodel of the
            # same group -- so the row above is the mechanism half of it and this one is
            # the second outcome half. Reported apart for the same reason: the influence
            # curve's mean is zero only if all three are solved, and says nothing about
            # which one is not.
            reduction = fluctuation.reduction
            if reduction is not None and np.asarray(reduction.score).size:
                score = float(np.max(np.abs(reduction.score)))
                stem = f"{group} (reduced)"
                rows.append(
                    ScoreCheckRow(
                        name=stem if result.n_repeats == 1 else f"{stem}[draw {index}]",
                        kind="fluctuation",
                        score=score,
                        threshold=float(threshold),
                        std_error=float(reference_se),
                        passed=bool(score <= threshold),
                        converged=reduction.converged,
                        n_iter=reduction.n_outer,
                        method=fluctuation.method,
                        score_initial=(
                            float(np.max(np.abs(reduction.score_initial)))
                            if np.asarray(reduction.score_initial).size
                            else float("nan")
                        ),
                        failure=reduction.failure or "",
                    )
                )

    # Each arm's corrections, recomputed from the state its draw returned. Two kinds of row
    # and they are two different failures: an ``identity`` row says the solver and the curve
    # are not evaluating one expression, which is a software defect; a ``correction`` row
    # says the term the curve subtracts is not negligible, which is an unsolved equation.
    # `cleverly.validation.drtmle` derives both, and returns nothing at all for a fit that
    # reports no corrections -- so no ordinary report gains a row.
    corrections = correction_check(result, tolerance=tolerance, std_error=float(reference_se))
    rows.extend(_correction_rows(corrections, n_repeats=result.n_repeats))

    for name, estimate in result.estimates.items():
        threshold = tolerance * estimate.std_error / np.sqrt(n)
        score = abs(estimate.score)
        rows.append(
            ScoreCheckRow(
                name=name,
                kind="influence curve",
                score=float(score),
                threshold=float(threshold),
                std_error=estimate.std_error,
                passed=bool(score <= threshold),
                converged=True,
                n_iter=0,
                method=estimate.scale,
            )
        )

    return ScoreCheck(
        rows=tuple(rows),
        tolerance=tolerance,
        n=n,
        # Read off the records rather than passed down from the estimator: this function
        # takes a result and a result does not name the class that made it.  A guard of
        # ``()`` leaves no reduction record, so such a fit is *not* corrected here -- which
        # is right, since it is a plain TMLE in every other respect too.
        corrected=any(
            fluctuation.reduction is not None
            for repeat in result.repeats
            for fluctuation in repeat.fluctuations.values()
        ),
        corrections=corrections,
    )


def _correction_rows(check: CorrectionCheck, *, n_repeats: int) -> list[ScoreCheckRow]:
    """One row per arm per equation, and a second where there is an identity to check.

    The draw index goes into the name only when there is more than one, exactly as the
    fluctuation rows do it, so an ordinary doubly-robust report reads as one fit's table.
    """
    rows: list[ScoreCheckRow] = []
    for row in check.rows:
        suffix = "" if n_repeats == 1 else f"[draw {row.draw}]"
        rows.append(
            ScoreCheckRow(
                name=f"{row.name}{suffix}",
                kind="correction",
                score=row.reported,
                threshold=check.threshold,
                std_error=check.std_error,
                passed=bool(abs(row.reported) <= check.threshold),
                converged=True,
                n_iter=0,
                method="recomputed",
                failure=(
                    ""
                    if row.solved
                    else "the reported curve subtracts this and the fit solved no equation "
                    "for it -- check the guard= this fit was given"
                ),
            )
        )
        if not np.isfinite(row.residual):
            # No stored score to compare against is the *absence* of a check, not a pass,
            # and the `failure` above is what says so.
            continue
        rows.append(
            ScoreCheckRow(
                name=f"{row.name} identity{suffix}",
                kind="identity",
                score=row.residual,
                threshold=check.identity_threshold,
                std_error=check.std_error,
                passed=bool(abs(row.residual) <= check.identity_threshold),
                converged=True,
                n_iter=0,
                method="recomputed",
                failure=(
                    ""
                    if abs(row.residual) <= check.identity_threshold
                    else f"the mechanism truncation binds on {row.clipped} row(s) and "
                    f"absorbs B_clip = {row.clip_bias:.3e} of this equation"
                    if np.isfinite(row.clip_bias)
                    else "the solved score and the reported term are different expressions"
                ),
            )
        )
    return rows
