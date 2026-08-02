"""The ``result.validation`` facade."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .drtmle import IDENTITY_TOLERANCE, CorrectionCheck, correction_check
from .nuisance import NuisanceDiagnostics, nuisance_diagnostics
from .refute import DEFAULT_TESTS, RefutationResult, refute
from .score import DEFAULT_TOLERANCE, ScoreCheck, score_check

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult

__all__ = ["ValidationSuite"]


class ValidationSuite:
    """Validation diagnostics for a fitted TMLE.

    Reached as ``result.validation``.  Run in increasing order of cost:

    ``score_check()``
        Free.  Did targeting solve the score equations this fit posed -- the estimated
        efficient one, or a doubly-robust fit's own three?  If this fails, nothing else is
        worth reading -- though passing it is necessary rather than sufficient, for the
        reason set out in :mod:`cleverly.validation.score`.  Also reached as
        ``result.score_verdict``, which is what ``result.summary()`` reports from.
    ``correction_check()``
        Free, and empty unless this is a guarded :class:`~cleverly.DRTMLE` fit.  Are the
        corrections the reported curve subtracts the ones the targeting step solved for,
        arm by arm?  Its verdict is already inside ``score_check()``; this is the
        recomputation itself -- see :mod:`cleverly.validation.drtmle`.
    ``nuisance()``
        Free.  How good are the initial fits, and are their probabilities calibrated?
    ``refute()``
        Several refits.  Does the estimate behave correctly on problems whose answer is
        known -- placebo treatment, added noise, subsamples?

    For validating an estimator *configuration* rather than a single fit, use
    :class:`~cleverly.validation.CoverageStudy`.
    """

    def __init__(self, result: TMLEResult) -> None:
        self._result = result

    def score_check(self, *, tolerance: float = DEFAULT_TOLERANCE) -> ScoreCheck:
        """Verify that the targeting step solved the score equations this fit posed.

        The estimated efficient score equation on an ordinary fit; a doubly-robust fit's
        own three, whose solution leaves the corrected curve rather than the efficient
        one -- see :mod:`cleverly.validation.score`.
        """
        return score_check(self._result, tolerance=tolerance)

    def correction_check(
        self,
        *,
        tolerance: float = DEFAULT_TOLERANCE,
        identity_tolerance: float = IDENTITY_TOLERANCE,
    ) -> CorrectionCheck:
        """Per arm, is the correction the curve subtracts the one the fit solved for?

        Free, and empty for every fit that reports no corrections.  ``score_check()``
        already reports the verdict; this is the recomputation behind it -- each arm's
        stored score, the mean of the term the reported curve carries, their difference,
        and the clipping bias that explains it.  See :mod:`cleverly.validation.drtmle`.
        """
        return correction_check(
            self._result, tolerance=tolerance, identity_tolerance=identity_tolerance
        )

    def nuisance(self) -> NuisanceDiagnostics:
        """Out-of-fold fit quality and calibration for every nuisance model."""
        return nuisance_diagnostics(self._result)

    def refute(
        self,
        *,
        estimand: str = "ate",
        tests: Sequence[str] = DEFAULT_TESTS,
        n_replicates: int = 5,
        subset_fraction: float = 0.7,
        negative_control_outcome: Any = None,
        random_state: int | None = None,
        tolerance: float = 3.0,
    ) -> RefutationResult:
        """Perturb the problem in ways whose correct answer is known, and check it."""
        return refute(
            self._result,
            estimand=estimand,
            tests=tests,
            n_replicates=n_replicates,
            subset_fraction=subset_fraction,
            negative_control_outcome=negative_control_outcome,
            random_state=random_state,
            tolerance=tolerance,
        )

    def report(self) -> str:
        """The two free diagnostics as one printable report."""
        return "\n\n".join([self.score_check().summary(), self.nuisance().summary()])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        check = self.score_check()
        return f"ValidationSuite(score_check={'pass' if check.passed else 'FAIL'})"
