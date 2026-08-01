"""The ``result.sensitivity`` facade.

Groups the sensitivity analyses behind one object so they are discoverable from a
fitted result, and so each one gets the cached nuisance fits without the caller
having to thread them through.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..interventions import (
    IncrementalSupport,
    ShiftSupport,
    SupportReport,
    check_incremental_support,
    check_shift_support,
    check_support,
)
from ._parameters import arm_parameters
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

    ``positivity()`` / ``support()`` / ``shift_support()`` / ``truncation_curve()``
        Can the *data* support the estimate?  Diagnoses overlap, not confounding.  Which
        of the three reports applies is set by the fit's parameter axis: arms, declared
        regimes, or declared shifts.
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

    def support(self) -> SupportReport:
        """Overlap *for the declared regimes*, which the arm-level table cannot show.

        A rule's positivity question is about ``g(d(W) | W)`` -- the propensity at the arm
        the rule assigns -- and two fits with identical marginal overlap can differ
        completely on it.  Raises on an arm-indexed fit, where there is no regime to
        report and :meth:`positivity` is the diagnostic.
        """
        regimes = self._result.nuisance.regimes
        if regimes is None:
            if self._result.nuisance.incremental is not None:
                raise ValueError(
                    "support() reports overlap for declared regimes; this fit declared "
                    "incremental interventions, whose overlap question is a different "
                    "one -- the clever covariate is bounded by max(delta, 1/delta) "
                    "whatever the mechanism does. Use incremental_support()."
                )
            raise ValueError(
                "support() reports overlap for the regimes a fit declared, and this fit "
                "declared none. Pass interventions= to TMLE, or use positivity() for the "
                "arm-level report."
            )
        return check_support(
            regimes, self._result.data.treatment, self._result.nuisance.propensity.values
        )

    def shift_support(self) -> dict[str, ShiftSupport]:
        """Overlap *for the declared shifts*, which is a question about a density ratio.

        A shift's clever covariate is ``g(a - delta | W) / g(a | W)``, so what threatens
        it is not an arm probability near zero but a *ratio* that runs away -- which
        happens where the shifted dose sits in the thin tail of the density that produced
        the observed one.  The report gives, per shift, the smallest density it divides
        by, the ratio's upper quantiles, the effective sample size those weights leave,
        and how many rows the cap held back.

        With ``delta=`` or ``intermediate=`` the covariate divides by a mechanism as well,
        and the report is of the whole weight ``h / (pi * q_z)``: the two reweightings
        multiply, so a ratio-only effective sample size would understate the strain.  The
        bound on that mechanism is ``nuisance_bound=`` and
        :meth:`truncation_curve` sweeps it.

        Raises on an arm-indexed or regime-indexed fit, where :meth:`positivity` and
        :meth:`support` are the diagnostics.
        """
        nuisance = self._result.nuisance
        shifts = nuisance.shifts
        density = nuisance.density
        if shifts is None or density is None:
            raise ValueError(
                "shift_support() reports overlap for the shifts a fit declared, and this "
                "fit declared none. Pass shifts= to TMLE, or use positivity() for the "
                "arm-level report and support() for a regime fit."
            )
        # The *bounded* mechanisms, because the bound is what the covariate actually
        # divided by; the raw arrays would report a strain the fit did not take.
        bound = self._result.config.missingness_bound
        level = self._result.intermediate_value
        mechanisms = [
            values
            for values in (
                nuisance.bounded_missingness(bound),
                None if level is None else nuisance.intermediate_density(level, bound),
            )
            if values is not None
        ]
        return check_shift_support(
            shifts, density, self._result.data.treatment, mechanisms=mechanisms
        )

    def incremental_support(self) -> dict[str, IncrementalSupport]:
        """Overlap *for the declared tilts*, which is a question with a known answer.

        Every other report on this class answers "how bad is the overlap"; this one
        answers "and why does it not matter here".  An incremental intervention's clever
        covariate is ``delta / D`` at ``A = 1`` and ``1 / D`` at ``A = 0``, so it lies
        between ``min(delta, 1/delta)`` and ``max(delta, 1/delta)`` however small the
        mechanism gets -- a bound the *analyst* chose rather than one the data granted.
        The report gives that bound beside the realised maximum, the smallest fitted
        propensity for contrast, and the effective sample size, which stays near ``n``
        where an arm-indexed fit's would have collapsed.

        What poor overlap threatens here is not the weights but the *consistency* of
        ``ghat``, on which this estimand depends with no doubly-robust fallback -- so
        :meth:`positivity` remains the diagnostic that matters, not this one.

        Raises on any other axis, where :meth:`positivity`, :meth:`support` and
        :meth:`shift_support` are the diagnostics.
        """
        tilts = self._result.nuisance.incremental
        if tilts is None:
            raise ValueError(
                "incremental_support() reports overlap for the tilts a fit declared, and "
                "this fit declared none. Pass incremental= to TMLE, or use positivity() "
                "for the arm-level report, support() for a regime fit and "
                "shift_support() for a shift fit."
            )
        return check_incremental_support(tilts, self._result.data.treatment)

    def truncation_curve(
        self,
        bounds: Sequence[float] | None = None,
        *,
        estimands: Sequence[str] | None = None,
        mechanism: bool = False,
    ) -> Any:
        """Estimates across a grid of truncation bounds.

        Sweeps the propensity bound by default, or -- with ``mechanism=True`` -- the
        bound on ``P(Delta = 1 | A, W)`` and the intermediate density, which divide the
        clever covariate for the same reason and deserve the same scrutiny.

        The sweep rests on a claim that holds for every axis but one: truncating a
        mechanism trades variance for second-order bias and cannot move the *estimand*,
        because the plug-in contains no mechanism at all.  On an incremental fit the
        *propensity* sweep does move the estimand -- ``q_delta`` is built out of ``g`` --
        so that one is refused rather than reported as a bias-variance trade it is not.

        The ``mechanism=True`` sweep is a different question and is allowed there.
        ``P(Delta = 1 | A, W)`` is not in ``q_delta`` and not in the plug-in; it divides
        the outcome half of the clever covariate and nothing else, so bounding it
        regularises the estimator on exactly the terms every other axis enjoys.  Refusing
        it along with the propensity sweep would have been refusing a valid analysis for a
        reason that does not apply to it.
        """
        if self._result.nuisance.incremental is not None and not mechanism:
            raise ValueError(
                "truncation_curve() sweeps the propensity bound, which on every other "
                "axis trades variance for second-order bias and leaves the estimand "
                "alone. On an incremental fit g is *inside* the estimand -- q_delta = "
                "delta*g / (delta*g + 1 - g) -- so each bound would target a different "
                "parameter and the curve would not be a sensitivity analysis. No bound "
                "is applied to g on this fit and none is needed: the clever covariate is "
                "bounded by max(delta, 1/delta) by construction. See "
                "incremental_support(). With delta= the missingness mechanism *is* an "
                "ordinary denominator and mechanism=True sweeps it."
            )
        return truncation_curve(self._result, bounds, estimands=estimands, mechanism=mechanism)

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
        """Bias-adjusted bounds under an assumed strength of unmeasured confounding.

        One bound per contrast: with more than two arms the estimand to name is
        ``"ate[medium vs low]"``, since the bound's ``nu2`` is the second moment of that
        contrast's own Riesz representer.
        """
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
        """VanderWeele--Ding E-value for the point estimate and confidence limit.

        One per contrast: with more than two arms the estimand to name is
        ``"rr[medium vs low]"``, and ``None`` picks the first ratio the fit reported.
        """
        return evalue(self._result, estimand)

    # ------------------------------------------------------------ missingness

    def missingness_tilt(
        self,
        gamma: Sequence[float] | None = None,
        *,
        estimands: Sequence[str] | None = None,
        arm_gamma: Mapping[Any, float] | None = None,
    ) -> Any:
        """Estimates under departures from missingness-at-random.

        ``arm_gamma=`` declares one multiplier per arm when the departure should not be
        assumed the same in each; the grid then sweeps that direction's magnitude.
        """
        return missingness_tilt(self._result, gamma, estimands=estimands, arm_gamma=arm_gamma)

    def tipping_gamma(
        self,
        estimand: str = "ate",
        *,
        null_hypothesis: float = 0.0,
        use_ci: bool = False,
        arm_gamma: Mapping[Any, float] | None = None,
    ) -> float | None:
        """The departure from MAR at which the conclusion would tip."""
        return tipping_gamma(
            self._result,
            estimand,
            null_hypothesis=null_hypothesis,
            use_ci=use_ci,
            arm_gamma=arm_gamma,
        )

    # --------------------------------------------------------------- combined

    def _reportable(self, estimand: str) -> str | None:
        """The parameter the report should be about, or ``None`` if there is none.

        ``"ate"`` names a parameter on a two-armed fit and none at all on a wider one,
        where the contrasts are ``ate[medium vs low]`` -- so a default that was a name
        has to become a *rule*: the requested one where it exists, else the first
        arm-indexed linear parameter the fit reported, in report order.
        """
        if estimand in self._result.estimates:
            return estimand
        known = arm_parameters(self._result.data, self._result.config.reference_arm)
        return next((name for name in self._result.estimates if name in known), None)

    def report(self, estimand: str = "ate") -> str:
        """Everything that can be computed without a refit, as one printable report."""
        name = self._reportable(estimand)
        blocks = [self.positivity().summary()]
        # Both analyses are scale-dependent and legitimately unavailable for some
        # fits (a ratio estimand, an outcome with no variance); skip rather than fail.
        if name is not None:
            with contextlib.suppress(ValueError):
                blocks.append(self.omitted_variable(name).summary())
        with contextlib.suppress(ValueError):
            blocks.append(self.evalue().summary())
        if self._result.data.has_missing_outcome and name is not None:
            tipping = self.tipping_gamma(name)
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
