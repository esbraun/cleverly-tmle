"""The assessment family of a continuous-time longitudinal fit (X6 part (e)).

:class:`~cleverly.survival.ContinuousLongitudinalResult` declares the family
``"continuous_longitudinal"``. Its support, score and nuisance diagnostics read the engine's own
record (:mod:`cleverly.survival.longitudinal_diagnostics`), and its influence report reads the
estimates. No adapter replays or refits the node passes, so the refutation and the truncation
curve refuse with an implementation gap on roadmap X105 part (d). The sensitivity analyses of
:mod:`cleverly.sensitivity` bound the effect of one point exposure, and a sustained regime over
decision visits is not one, so each refuses with an inference gap. The capability rows declare
the same records, so a combined report names each refusal and never raises it.

A leaf module: it imports :mod:`cleverly._refusal` only, and :mod:`cleverly.exceptions` inside
:func:`refuse_continuous_longitudinal`.
"""

from __future__ import annotations

from typing import Any, Final

from ._refusal import NO_ALTERNATIVE, Refusal, RefusalReason

__all__ = [
    "CONTINUOUS_LONGITUDINAL_FAMILY",
    "SCORE_TOLERANCE_REFUSAL",
    "continuous_longitudinal_refusal",
    "is_continuous_longitudinal",
    "refuse_continuous_longitudinal",
    "replay_refusal",
]

#: The assessment family of a continuous-time longitudinal fit.
CONTINUOUS_LONGITUDINAL_FAMILY: Final = "continuous_longitudinal"

#: The refits and replays: no adapter replays a node pass (X105 part (d)).
REPLAY_TEXT: Final[str] = (
    "{operation} replays or refits a fit through an adapter, and no adapter replays the node "
    "passes of the continuous-time longitudinal TMLE."
)

#: The sensitivity analyses bound one point exposure's effect.
SENSITIVITY_TEXT: Final[str] = (
    "The sensitivity analyses of this package bound the effect of one point exposure, and a "
    "sustained regime over decision visits is not one: an E-value reads one point exposure's "
    "risk ratio."
)

#: What a caller can read instead of a sensitivity analysis.
SENSITIVITY_ALTERNATIVE: Final[str] = (
    "Read the regimes' risks and their differences with their intervals."
)

#: An explicit tolerance on ``score_equations``: the node walks' own cutoff gates the report.
SCORE_TOLERANCE_REFUSAL: Final = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "score_equations(tolerance=...) does not apply to a continuous-time longitudinal fit: each "
    "node walk stops at s_n / N, with s_n = 1 / (sqrt(n) log n) or the tol the fit was given, "
    "and the report gates every target's |P_n D*| / sigma on s_n. A caller-chosen tolerance "
    "would not change what the fit solved.",
    alternative="Call score_equations() with no tolerance and read the report's cutoff.",
)

#: The correction identities belong to guarded doubly-robust point-treatment targeting.
_CORRECTIONS: Final = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "the continuous-time longitudinal TMLE does not use the correction system",
)

#: A report a caller reached as a free function rather than the facade.
_REPORT: Final = Refusal(
    RefusalReason.API_REDIRECTION,
    "this report reads a point-treatment or discrete-time fit's state, which a "
    "continuous-time longitudinal fit does not hold.",
    alternative=(
        "Read result.diagnostics.support(), score_equations(), nuisance_models() and "
        "influence(), which report the continuous-time longitudinal fit's own record."
    ),
)


def replay_refusal(operation: str) -> Refusal:
    """The refusal of a refit or a replay of a continuous-time longitudinal fit.

    Parameters
    ----------
    operation : str
        The operation, in words, such as ``"refute()"``.

    Returns
    -------
    Refusal
        An ``implementation_gap`` record naming ``X105``.

    Examples
    --------
    >>> from cleverly._continuous_longitudinal_family import replay_refusal
    >>> replay_refusal("refute()").roadmap
    'X105'
    """
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        REPLAY_TEXT.format(operation=operation),
        alternative=NO_ALTERNATIVE,
        roadmap="X105",
    )


#: The sensitivity refusal, shared by every sensitivity row.
_SENSITIVITY: Final = Refusal(
    RefusalReason.INFERENCE_GAP, SENSITIVITY_TEXT, alternative=SENSITIVITY_ALTERNATIVE
)

#: The diagnostics rows that refit or replay, by operation.
_REPLAYED: Final[dict[str, str]] = {
    "refute": "refute()",
    "refute_estimate": "refute()",
    "truncation_curve": "truncation_curve()",
    "continuous_truncation_curve": "continuous_truncation_curve()",
}

#: The sensitivity rows, each refused with :data:`_SENSITIVITY`.
_SENSITIVITY_ROWS: Final = frozenset(
    {
        "omitted_confounding",
        "robustness_value",
        "elements",
        "contour",
        "benchmark",
        "marginal_bounds",
        "missingness",
        "tipping_gamma",
        "simulated_confounding",
        "evalue",
        "omitted_variable_bounds",
        "sensitivity_elements",
        "contour_data",
        "marginal_sensitivity_bounds",
        "missingness_tilt",
        "e_value",
    }
)


def continuous_longitudinal_refusal(operation: str) -> Refusal:
    """The refusal of one analysis on a continuous-time longitudinal fit.

    Parameters
    ----------
    operation : str
        A capability row's operation, or the free function that implements it.

    Returns
    -------
    Refusal
        ``implementation_gap`` on ``X105`` for a refit or a replay, ``inference_gap`` for a
        sensitivity analysis, ``invalid_declaration`` for the correction identities and
        ``api_redirection`` for any other report.

    Examples
    --------
    >>> from cleverly._continuous_longitudinal_family import continuous_longitudinal_refusal
    >>> continuous_longitudinal_refusal("refute").roadmap
    'X105'
    >>> continuous_longitudinal_refusal("evalue").reason.value
    'inference_gap'
    """
    if operation in _REPLAYED:
        return replay_refusal(_REPLAYED[operation])
    if operation in _SENSITIVITY_ROWS:
        return _SENSITIVITY
    if operation in ("corrections", "correction_check"):
        return _CORRECTIONS
    return _REPORT


def is_continuous_longitudinal(result: Any) -> bool:
    """Whether ``result`` declares the ``"continuous_longitudinal"`` assessment family.

    Parameters
    ----------
    result : Any
        A fitted result.

    Returns
    -------
    bool
        ``True`` for a continuous-time longitudinal fit.
    """
    return getattr(result, "assessment_family", None) == CONTINUOUS_LONGITUDINAL_FAMILY


def refuse_continuous_longitudinal(result: Any, operation: str) -> None:
    """Raise :func:`continuous_longitudinal_refusal` for a continuous-time longitudinal fit.

    Parameters
    ----------
    result : Any
        A fitted result.
    operation : str
        The analysis the caller asked for.

    Raises
    ------
    CapabilityError
        When ``result`` declares the ``"continuous_longitudinal"`` family.
    """
    if is_continuous_longitudinal(result):
        from .exceptions import CapabilityError

        raise CapabilityError.from_refusal(continuous_longitudinal_refusal(operation))
