"""Shared provenance checks for simulated-confounding replay axes."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import CapabilityError, DataError
from ..study import PointTreatment
from ..targets import TARGETS
from ..targets.base import stratum_alias

_AXIS_SLOTS = {
    "arm": (),
    "shift": ("shifts",),
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


def check_registered_target(result: Any, key: Any, axis: str, error: str) -> Any:
    """Require the functional, key, fitted configuration, and registry to agree."""
    identified = result.identified_effect
    registered = TARGETS.get(key.estimand)
    if (
        registered is None
        or identified.functional.target != key.estimand
        or identified.identification != registered.identification
        or registered.parameter_axis != axis
        or identified.functional.axis != axis
        or key.axis != axis
        or result.config.parameter_axis != axis
        or key.estimand not in result.config.estimands
    ):
        raise CapabilityError(error)
    return registered


def check_replay_declaration(result: Any, key: Any, error: str) -> None:
    """Require the replay estimator to declare the fitted target and reference."""
    estimator = result.estimator
    if (
        key.estimand not in tuple(estimator.estimands or ())
        or estimator.reference != result.identified_effect.functional.reference
    ):
        raise CapabilityError(error)


def check_alias(
    result: Any,
    estimand: str,
    key: Any,
    alias: str,
    error: str,
    *,
    estimate_error: str | None = None,
    finite_error: str | None = None,
) -> None:
    """Require an alias and its stored estimate to identify the requested parameter."""
    estimate = result.estimates[estimand]
    if key.alias != estimand or _population_alias(result, key, alias) != estimand:
        raise CapabilityError(error)
    if estimate.name != estimand:
        raise CapabilityError(estimate_error or error)
    if not np.isfinite(estimate.psi):
        raise CapabilityError(finite_error or error)


def point_study_or_refuse(result: Any, error: str) -> Any:
    """Return the study after checking its point-treatment data contract."""
    identified = result.identified_effect
    study = getattr(identified, "_study", None)
    if study is None or type(study.design) is not PointTreatment:
        raise CapabilityError(error)
    try:
        study.design._check_prepared(result.data)
    except DataError as cause:
        # Only data-contract failures describe corrupted provenance. Programming errors
        # and user-callback exceptions must retain their original type and traceback.
        raise CapabilityError(f"{error}: {cause}") from cause
    return study


def check_only_declared_axis(result: Any, key: Any, axis: str, error: str) -> None:
    """Refuse compositions without a supported perturbation and replay law."""
    functional = result.identified_effect.functional
    estimator = result.estimator
    allowed = _AXIS_SLOTS[axis]
    slots = {slot for names in _AXIS_SLOTS.values() for slot in names}
    if (
        functional.longitudinal
        or functional.axis != axis
        or functional.horizons is not None
        or functional.intermediate is not None
        or any(getattr(estimator, slot) for slot in slots if slot not in allowed)
        or (bool(functional.interventions) != (axis in {"shift", "regime", "ipsi"}))
        or (functional.msm is not None) != (axis == "msm")
        or any(getattr(key, slot) is not None for slot in ("regimen", "cause", "horizon"))
        or (axis != "msm" and key.term is not None)
    ):
        raise CapabilityError(error)
