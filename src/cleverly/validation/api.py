"""The ``result.validation`` facade."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

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
        Free.  Did targeting solve the efficient score equation?  If this fails, nothing
        else is worth reading.
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
        """Verify that the targeting step solved the efficient score equation."""
        return score_check(self._result, tolerance=tolerance)

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
