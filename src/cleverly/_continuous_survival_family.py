"""The assessment family of a continuous-time survival fit.

:class:`~cleverly.survival.ContinuousSurvivalResult` declares the family
``"continuous_survival"``. Its diagnostics read the engine's own record: the hazard,
censoring and treatment selections, the standardized scores of the targeting walk, the
influence curves, and the treatment and censoring probabilities. The E-value reads the
fit's risk ratio. The refutation and the truncation curve refit the estimator the result
stores on its stored data (:mod:`cleverly.validation._continuous_survival_refute`,
:func:`cleverly.survival.diagnostics.continuous_truncation_curve`). The simulated common cause
and every other sensitivity analysis of :mod:`cleverly.sensitivity` read a point-treatment
result's nuisance fits or targeting state, or derive a bound for a mean of one outcome
regression. None of them is implemented or derived for a continuous-time risk, so each refuses
with the record :func:`continuous_survival_refusal` returns. The capability rows declare the
same records, so a combined report names each refusal and never raises it.

A leaf module: it imports :mod:`cleverly._refusal` only, and :mod:`cleverly.exceptions`
inside :func:`refuse_continuous_survival`.
"""

from __future__ import annotations

from typing import Any, Final

from ._refusal import NO_ALTERNATIVE, Refusal, RefusalReason

__all__ = [
    "CONTINUOUS_SURVIVAL_FAMILY",
    "SCORE_TOLERANCE_REFUSAL",
    "continuous_survival_refusal",
    "is_continuous_survival",
    "refuse_continuous_survival",
]

#: The assessment family of a continuous-time survival fit.
CONTINUOUS_SURVIVAL_FAMILY: Final = "continuous_survival"

#: The simulated common cause redraws event and censoring times under a latent shift.
_SIMULATED: Final = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "no simulated-confounding replay is implemented for a continuous-time survival fit: the "
    "replay redraws event and censoring times under a latent shift of the cause-specific "
    "hazards, and the shipped survival replay reads a time grid.",
    alternative="The placebo, random_common_cause and subset tests run on this fit.",
    roadmap="X101",
)

#: The omitted-variable bounds read the Riesz representer of a point-treatment functional.
_OMITTED: Final = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "no omitted-variable bound is derived for a continuous-time survival functional: the "
    "point-treatment bound reads the Riesz representer of a mean of one outcome regression, "
    "and no representer of a cause-specific absolute risk under censoring is derived.",
    alternative=NO_ALTERNATIVE,
    roadmap="F16",
)

#: The marginal sensitivity model bounds an arm mean of one outcome.
_MARGINAL: Final = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "no marginal sensitivity model bound is derived for a continuous-time survival "
    "functional: the point-treatment bound tilts the outcome regression of an arm mean, and "
    "no tilt of the cause-specific hazards of a censored absolute risk is derived.",
    alternative=NO_ALTERNATIVE,
    roadmap="F16",
)

#: The missingness tilt perturbs a missing-at-random outcome mechanism.
_TILT: Final = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "no missingness tilt is derived for a continuous-time survival fit: the tilt perturbs a "
    "missing-at-random outcome mechanism, the fit's observation mechanism is independent "
    "censoring in continuous time, and no tilt of that assumption is derived.",
    alternative=NO_ALTERNATIVE,
    roadmap="F16",
)

#: An explicit tolerance on ``score_equations``: the walk's own cutoff gates the report.
SCORE_TOLERANCE_REFUSAL: Final = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "score_equations(tolerance=...) does not apply to a continuous-time survival fit: the "
    "walk stops when every |P_n D*| / sigma is at or below 1 / (sqrt(n) log n), or the tol "
    "the fit was given, and the report gates on that cutoff. A caller-chosen tolerance would "
    "not change what the fit solved.",
    alternative="Call score_equations() with no tolerance and read the report's cutoff.",
)

#: The correction identities belong to guarded doubly-robust point-treatment targeting.
_CORRECTIONS: Final = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "the continuous-time survival TMLE does not use the correction system",
)

#: A point-treatment report a caller reached as a free function rather than the facade.
_REPORT: Final = Refusal(
    RefusalReason.API_REDIRECTION,
    "this report reads a point-treatment fit's nuisance and targeting state, which a "
    "continuous-time survival fit does not hold.",
    alternative=(
        "Read result.diagnostics.support(), score_equations(), nuisance_models() and "
        "influence(), which report the continuous-time fit's own record."
    ),
)

#: The refusal of each refused operation, by the name its capability row declares.
_BY_OPERATION: Final[dict[str, Refusal]] = {
    "simulated_confounding": _SIMULATED,
    "omitted_confounding": _OMITTED,
    "robustness_value": _OMITTED,
    "elements": _OMITTED,
    "contour": _OMITTED,
    "benchmark": _OMITTED,
    "marginal_bounds": _MARGINAL,
    "missingness": _TILT,
    "tipping_gamma": _TILT,
    "corrections": _CORRECTIONS,
}

#: The free functions of :mod:`cleverly.sensitivity` and :mod:`cleverly.validation`, by the
#: capability row whose refusal each raises.
_FUNCTIONS: Final[dict[str, str]] = {
    "omitted_variable_bounds": "omitted_confounding",
    "sensitivity_elements": "elements",
    "contour_data": "contour",
    "marginal_sensitivity_bounds": "marginal_bounds",
    "missingness_tilt": "missingness",
    "correction_check": "corrections",
}


def continuous_survival_refusal(operation: str) -> Refusal:
    """The refusal of one analysis on a continuous-time survival fit.

    Parameters
    ----------
    operation : str
        A capability row's operation, or the free function that implements it, such as
        ``"simulated_confounding"``, ``"omitted_confounding"`` or
        ``"omitted_variable_bounds"``. A name that no refused row covers is a point-treatment
        report.

    Returns
    -------
    Refusal
        An ``implementation_gap`` record naming ``X101`` for the simulated common cause and
        ``F16`` for the omitted-variable, marginal and tilt bounds; an ``invalid_declaration``
        record for the correction identities, which the fit does not use; an
        ``api_redirection`` record for a point-treatment report.

    Examples
    --------
    >>> from cleverly._continuous_survival_family import continuous_survival_refusal
    >>> continuous_survival_refusal("simulated_confounding").roadmap
    'X101'
    >>> continuous_survival_refusal("omitted_variable_bounds").roadmap
    'F16'
    """
    return _BY_OPERATION.get(_FUNCTIONS.get(operation, operation), _REPORT)


def is_continuous_survival(result: Any) -> bool:
    """Whether ``result`` declares the ``"continuous_survival"`` assessment family.

    Parameters
    ----------
    result : Any
        A fitted result.

    Returns
    -------
    bool
        ``True`` for a continuous-time survival fit.
    """
    return getattr(result, "assessment_family", None) == CONTINUOUS_SURVIVAL_FAMILY


def refuse_continuous_survival(result: Any, operation: str) -> None:
    """Raise the refusal of :func:`continuous_survival_refusal` for a continuous-time fit.

    Parameters
    ----------
    result : Any
        A fitted result.
    operation : str
        The analysis the caller asked for.

    Raises
    ------
    CapabilityError
        When ``result`` declares the ``"continuous_survival"`` family.
    """
    if is_continuous_survival(result):
        from .exceptions import CapabilityError

        raise CapabilityError.from_refusal(continuous_survival_refusal(operation))
