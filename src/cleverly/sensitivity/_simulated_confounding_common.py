"""Shared provenance checks for simulated-confounding replay axes."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._refusal import Refusal, RefusalReason
from ..exceptions import CapabilityError, DataError
from ..study import PointTreatment, _matches_registered_point_identification
from ..targets import TARGETS
from ..targets.base import stratum_alias

#: A caller's refusal text: a plain string states an invalid declaration, and a record states
#: any other code with its alternative.
ReplayError = str | Refusal


def replay_error(error: ReplayError) -> CapabilityError:
    """Build the refusal one replay check raises.

    Parameters
    ----------
    error : str or Refusal
        A string states an inconsistent-metadata refusal, which is an invalid declaration.
        A record carries its own code.

    Returns
    -------
    CapabilityError
        The error to raise.
    """
    if isinstance(error, Refusal):
        return CapabilityError.from_refusal(error)
    return CapabilityError(error, reason=RefusalReason.INVALID_DECLARATION)


_AXIS_SLOTS = {
    "arm": (),
    "policy": ("policies",),
    "rr_tilt": ("policies",),
    "regime": ("interventions",),
    "msm": ("msm",),
    "ipsi": ("incremental",),
}


def fixed_axis(target: str) -> str | None:
    """Return the audited fixed-policy axis."""
    return {"ey_regime": "regime", "ate_regime": "regime", "msm": "msm"}.get(target)


def fixed_key_axis(key: Any) -> str | None:
    """Return a fixed axis only when the structured key agrees."""
    axis = fixed_axis(key.estimand)
    return axis if key.axis == axis else None


def _population_alias(result: Any, key: Any, marginal_alias: str) -> str:
    """Compose the selected alias without parsing caller-owned labels."""
    if key.stratum is None:
        return marginal_alias
    code = result.data.strata_levels.index(key.stratum)
    return stratum_alias(marginal_alias, result.data.stratum_label(code))


def check_registered_target(result: Any, key: Any, axis: str, error: ReplayError) -> Any:
    """Require the functional, key, fitted configuration, and registry to agree."""
    identified = result.identified_effect
    registered = TARGETS.get(key.estimand)
    if (
        registered is None
        or identified.functional.target != key.estimand
        or not _matches_registered_point_identification(
            identified.identification,
            registered.identification,
            identified,
            result.data,
            axis,
        )
        or registered.parameter_axis != axis
        or identified.functional.axis != axis
        or key.axis != axis
        or result.config.parameter_axis != axis
        or key.estimand not in result.config.estimands
    ):
        raise replay_error(error)
    return registered


def check_replay_declaration(result: Any, key: Any, error: ReplayError) -> None:
    """Require the replay estimator to declare the fitted target and reference."""
    estimator = result.estimator
    if (
        key.estimand not in tuple(estimator.estimands or ())
        or estimator.reference != result.identified_effect.functional.reference
    ):
        raise replay_error(error)


def check_alias(
    result: Any,
    estimand: str,
    key: Any,
    alias: str,
    error: ReplayError,
    *,
    estimate_error: ReplayError | None = None,
    finite_error: ReplayError | None = None,
) -> None:
    """Require an alias and its stored estimate to identify the requested parameter."""
    estimate = result.estimates[estimand]
    if key.alias != estimand or _population_alias(result, key, alias) != estimand:
        raise replay_error(error)
    if estimate.name != estimand:
        raise replay_error(estimate_error or error)
    if not np.isfinite(estimate.psi):
        raise replay_error(finite_error or error)


def point_study_or_refuse(result: Any, error: ReplayError) -> Any:
    """Return the study after checking its point-treatment data contract."""
    identified = result.identified_effect
    study = getattr(identified, "_study", None)
    if study is None or type(study.design) is not PointTreatment:
        raise replay_error(error)
    try:
        study.design._check_prepared(result.data)
    except DataError as cause:
        # Only data-contract failures describe corrupted provenance. Programming errors
        # and user-callback exceptions must retain their original type and traceback.
        # A record's condition ends with a full stop, and the join adds a colon.
        context = error.condition.rstrip(".") if isinstance(error, Refusal) else error
        raise CapabilityError(
            f"{context}: {cause.refusal.condition}",
            reason=cause.refusal.reason,
            alternative=cause.refusal.alternative,
            roadmap=cause.refusal.roadmap,
        ) from cause
    return study


def check_only_declared_axis(
    result: Any, key: Any, axis: str, error: ReplayError, *, intermediate: bool = False
) -> None:
    """Refuse compositions without a supported perturbation and replay law.

    ``intermediate=True`` admits a controlled direct effect on the arm axis, whose replay
    holds the treatment and perturbs the intermediate and the outcome (X82 part F (f)).
    """
    functional = result.identified_effect.functional
    estimator = result.estimator
    allowed = _AXIS_SLOTS[axis]
    slots = {slot for names in _AXIS_SLOTS.values() for slot in names}
    if (
        functional.longitudinal
        or functional.axis != axis
        or functional.horizons is not None
        or (functional.intermediate is not None and not (intermediate and axis == "arm"))
        or any(getattr(estimator, slot) for slot in slots if slot not in allowed)
        or (bool(functional.interventions) != (axis in {"policy", "regime", "ipsi"}))
        or (functional.msm is not None) != (axis == "msm")
        or any(getattr(key, slot) is not None for slot in ("regimen", "cause", "horizon"))
        or (axis != "msm" and key.term is not None)
    ):
        raise replay_error(error)
