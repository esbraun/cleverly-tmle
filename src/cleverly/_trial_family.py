"""The assessment family of an adaptive TMLE fit of a trial with external data.

:class:`~cleverly.estimators.atmle_result.AdaptiveTMLEResult` declares the family
``"augmented_trial"``. The sensitivity analyses and the refutation of
:mod:`cleverly.sensitivity` and :mod:`cleverly.validation` read a point-treatment result's
nuisance fits, its configuration and its targeting state. None of them is derived for an
adaptive working-model fit, so each entry point calls :func:`refuse_augmented_trial` first,
keyed on the declared family, and refuses before any point-treatment code runs.

A leaf module: it imports :mod:`cleverly._refusal` and :mod:`cleverly.exceptions` only.
"""

from __future__ import annotations

from typing import Any, Final

from ._refusal import NO_ALTERNATIVE, Refusal, RefusalReason

__all__ = [
    "AUGMENTED_TRIAL_FAMILY",
    "augmented_trial_refusal",
    "is_augmented_trial",
    "refuse_augmented_trial",
]

#: The assessment family of an adaptive TMLE fit of a trial with external data.
AUGMENTED_TRIAL_FAMILY: Final = "augmented_trial"

#: What an adaptive working-model fit lacks for each kind of analysis, keyed by the noun
#: the refusal names.
_MISSING: Final[dict[str, str]] = {
    "bound": (
        "no bound is derived for an adaptive working-model fit, whose estimand is the "
        "projection of the trial effect onto a working model the data selected"
    ),
    "refutation": (
        "no refutation is derived for an adaptive working-model fit: each perturbed refit "
        "selects its own working model, so the refits do not estimate the same projection"
    ),
    "truncation": (
        "no truncation curve is derived for an adaptive working-model fit: a refit at "
        "another treatment bound selects its own working model, so the points of the curve "
        "do not estimate the same projection"
    ),
    "report": (
        "this report reads a point-treatment fit's nuisance and targeting state, which an "
        "adaptive working-model fit does not hold"
    ),
}


def augmented_trial_refusal(operation: str, kind: str = "bound") -> Refusal:
    """The refusal of one analysis on an adaptive TMLE fit of a trial with external data.

    Parameters
    ----------
    operation : str
        The analysis the caller asked for, as the refusal names it.
    kind : {"bound", "refutation", "truncation", "report"}, default="bound"
        Which missing derivation the refusal states.

    Returns
    -------
    Refusal
        An ``inference_gap`` record for a bound, a refutation or a truncation curve, and an
        ``api_redirection`` record that points at the fit's own diagnostics for a report.
    """
    if kind == "report":
        return Refusal(
            RefusalReason.API_REDIRECTION,
            f"{operation}: {_MISSING[kind]}.",
            alternative=(
                "Read result.diagnostics.support(), score_equations(), nuisance_models() and "
                "corrections(), which report the adaptive fit's own state."
            ),
        )
    return Refusal(
        RefusalReason.INFERENCE_GAP,
        f"{operation}: {_MISSING[kind]}.",
        alternative=NO_ALTERNATIVE,
    )


def is_augmented_trial(result: Any) -> bool:
    """Whether ``result`` declares the ``"augmented_trial"`` assessment family.

    Parameters
    ----------
    result : Any
        A fitted result.

    Returns
    -------
    bool
        ``True`` for an adaptive TMLE fit of a trial with external data.
    """
    return getattr(result, "assessment_family", None) == AUGMENTED_TRIAL_FAMILY


def refuse_augmented_trial(result: Any, operation: str, kind: str = "bound") -> None:
    """Raise the refusal of :func:`augmented_trial_refusal` for an augmented-trial fit.

    Parameters
    ----------
    result : Any
        A fitted result.
    operation : str
        The analysis the caller asked for.
    kind : {"bound", "refutation", "truncation", "report"}, default="bound"
        Which missing derivation the refusal states.

    Raises
    ------
    CapabilityError
        When ``result`` declares the ``"augmented_trial"`` family.
    """
    if is_augmented_trial(result):
        from .exceptions import CapabilityError

        raise CapabilityError.from_refusal(augmented_trial_refusal(operation, kind))
