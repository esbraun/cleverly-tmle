"""The ``result.sensitivity`` facade.

Groups the sensitivity analyses behind one object so they are discoverable from a
fitted result, and so each one gets the cached nuisance fits without the caller
having to thread them through.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .evalue import EValue, evalue
from .missingness import missingness_tilt, tipping_gamma
from .omitted_variable import (
    BenchmarkResult,
    SensitivityBounds,
    SensitivityElements,
    benchmark,
    contour_data,
    omitted_variable_bounds,
    robustness_value,
    sensitivity_elements,
)
from .positivity import PositivityReport, positivity_report, truncation_curve

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult

__all__ = ["SensitivityAnalysis"]


class SensitivityAnalysis:
    """Sensitivity analyses for a fitted TMLE.

    Reached as ``result.sensitivity``.  The three families answer different questions
    and are worth running together:

    ``positivity()`` / ``truncation_curve()``
        Can the *data* support the estimate?  Diagnoses overlap, not confounding.
    ``omitted_variable()`` / ``robustness_value()`` / ``benchmark()`` / ``evalue()``
        How strong would unmeasured confounding have to be to change the conclusion?
    ``missingness_tilt()`` / ``tipping_gamma()``
        How much would the missing outcomes have to differ from the observed ones?
    """

    def __init__(self, result: TMLEResult) -> None:
        self._result = result

    # ------------------------------------------------------------- positivity

    def positivity(self) -> PositivityReport:
        """Overlap diagnostics: propensity distribution, effective sample size, weights."""
        return positivity_report(self._result)

    def truncation_curve(
        self, bounds: Sequence[float] | None = None, *, estimands: Sequence[str] | None = None
    ) -> Any:
        """Estimates across a grid of propensity-truncation bounds."""
        return truncation_curve(self._result, bounds, estimands=estimands)

    # ------------------------------------------------------ omitted variables

    def omitted_variable(
        self,
        estimand: str = "ate",
        *,
        cf_y: float = 0.03,
        cf_d: float = 0.03,
        rho: float = 1.0,
        level: float = 0.95,
        null_hypothesis: float = 0.0,
        nu2_estimator: str = "auto",
    ) -> SensitivityBounds:
        """Bias-adjusted bounds under an assumed strength of unmeasured confounding."""
        return omitted_variable_bounds(
            self._result,
            estimand,
            cf_y=cf_y,
            cf_d=cf_d,
            rho=rho,
            level=level,
            null_hypothesis=null_hypothesis,
            nu2_estimator=nu2_estimator,
        )

    def robustness_value(
        self,
        estimand: str = "ate",
        *,
        rho: float = 1.0,
        level: float = 0.95,
        null_hypothesis: float = 0.0,
    ) -> dict[str, float]:
        """Confounding strength that would explain the effect away -- no assumption needed."""
        return robustness_value(
            self._result,
            estimand,
            rho=rho,
            level=level,
            null_hypothesis=null_hypothesis,
        )

    def elements(
        self, estimand: str = "ate", *, nu2_estimator: str = "auto"
    ) -> SensitivityElements:
        """The raw ingredients of the bias bound, for custom analyses."""
        return sensitivity_elements(self._result, estimand, nu2_estimator=nu2_estimator)

    def benchmark(
        self, covariates: Sequence[str] | str, *, estimand: str = "ate"
    ) -> BenchmarkResult:
        """Calibrate the sensitivity parameters against observed covariates (refits)."""
        return benchmark(self._result, covariates, estimand=estimand)

    def contour(
        self,
        estimand: str = "ate",
        *,
        rho: float = 1.0,
        grid_size: int = 20,
        grid_bounds: tuple[float, float] = (0.15, 0.15),
        bound: str = "lower",
    ) -> Any:
        """Grid of bias-adjusted bounds over ``cf_d`` x ``cf_y``, for a contour plot."""
        return contour_data(
            self._result,
            estimand,
            rho=rho,
            grid_size=grid_size,
            grid_bounds=grid_bounds,
            bound=bound,
        )

    def evalue(self, estimand: str | None = None) -> EValue:
        """VanderWeele--Ding E-value for the point estimate and confidence limit."""
        return evalue(self._result, estimand)

    # ------------------------------------------------------------ missingness

    def missingness_tilt(
        self,
        gamma: Sequence[float] | None = None,
        *,
        estimands: Sequence[str] | None = None,
    ) -> Any:
        """Estimates under departures from missingness-at-random."""
        return missingness_tilt(self._result, gamma, estimands=estimands)

    def tipping_gamma(
        self,
        estimand: str = "ate",
        *,
        null_hypothesis: float = 0.0,
        use_ci: bool = False,
    ) -> float | None:
        """The departure from MAR at which the conclusion would tip."""
        return tipping_gamma(self._result, estimand, null_hypothesis=null_hypothesis, use_ci=use_ci)

    # --------------------------------------------------------------- combined

    def report(self, estimand: str = "ate") -> str:
        """Everything that can be computed without a refit, as one printable report."""
        blocks = [self.positivity().summary()]
        # Both analyses are scale-dependent and legitimately unavailable for some
        # fits (a ratio estimand, an outcome with no variance); skip rather than fail.
        if estimand in self._result.estimates:
            with contextlib.suppress(ValueError):
                blocks.append(self.omitted_variable(estimand).summary())
        with contextlib.suppress(ValueError):
            blocks.append(self.evalue().summary())
        if self._result.data.has_missing_outcome:
            tipping = self.tipping_gamma(estimand if estimand in self._result.estimates else "ate")
            blocks.append(
                "Missingness tilt\n"
                + "-" * 16
                + "\n"
                + (
                    f"the estimate reaches the null at gamma = {tipping:.3g}"
                    if tipping is not None
                    else "no tilt within [-8, 8] on the logit scale reaches the null"
                )
            )
        return "\n\n".join(blocks)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SensitivityAnalysis(estimands={list(self._result.estimates)})"
