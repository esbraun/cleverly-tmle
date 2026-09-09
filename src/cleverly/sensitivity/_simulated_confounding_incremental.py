"""Validate fixed odds multipliers and law-dependent incremental replay.

Kennedy (2019), Equation (1) and Corollaries 1--2, define the intervention through
the treatment mechanism. The declarations stay fixed; complete refits rebuild both
the mechanism and its tilted densities. No cached density is frozen across cells.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import CapabilityError
from ..interventions import Incremental, IPSISet
from ..study import IncrementalEffect, IncrementalMean, ParameterKey
from ..targets.base import parameter_name
from ._simulated_confounding_common import (
    check_alias,
    check_only_declared_axis,
    check_registered_target,
    check_replay_declaration,
    point_study_or_refuse,
)

_TARGET_TYPES = {"ey_ipsi": IncrementalMean, "ate_ipsi": IncrementalEffect}
INCREMENTAL_TARGETS = frozenset(_TARGET_TYPES)
_STRATA_REFUSAL = (
    "simulated_confounding cannot replay baseline strata for incremental targets; "
    "stratified alternating targeting is unsupported"
)


def incremental_replay_refusal(
    estimator: Any, target: str, stratum: tuple[Any, ...] | None = None
) -> str | None:
    """Describe the estimator boundary without evaluating a treatment mechanism."""
    from ..estimators.tmle import TMLE

    if target in INCREMENTAL_TARGETS and type(estimator) is not TMLE:
        return "simulated_confounding supports incremental targets under exact ordinary TMLE only"
    if target in INCREMENTAL_TARGETS and stratum is not None:
        return _STRATA_REFUSAL
    return None


def natural_incremental_means(result: Any) -> set[str]:
    """Identify declared natural-course means for the supported-alias filter."""
    identified = getattr(result, "identified_effect", None)
    functional = getattr(identified, "functional", None)
    declarations = getattr(functional, "interventions", ()) or ()
    natural = {
        item.name for item in declarations if type(item) is Incremental and item.delta == 1.0
    }
    return {
        alias
        for alias, key in (getattr(result, "parameter_keys", None) or {}).items()
        if type(key) is ParameterKey
        and key.estimand == "ey_ipsi"
        and key.axis == "ipsi"
        and key.value in natural
    }


def validate_incremental_replay(result: Any, estimand: str, key: Any) -> Any:
    """Check every declaration and draw before replaying the full incremental fit."""
    error = "simulated_confounding found inconsistent incremental parameter metadata"
    identified = result.identified_effect
    functional, typed, estimator = identified.functional, identified.estimand, result.estimator
    if type(typed) is not _TARGET_TYPES.get(key.estimand):
        raise CapabilityError(error)
    refusal = incremental_replay_refusal(estimator, key.estimand)
    if refusal is not None:
        raise CapabilityError(refusal)
    if result.data.has_strata:
        raise CapabilityError(_STRATA_REFUSAL)
    registered = check_registered_target(result, key, "ipsi", error)
    check_replay_declaration(result, key, error)
    check_only_declared_axis(result, key, "ipsi", error)
    study = point_study_or_refuse(result, error)
    declarations = tuple(functional.interventions)
    if (
        not declarations
        or any(type(item) is not Incremental for item in declarations)
        or any(type(item) is not Incremental for item in estimator.incremental)
        or any(type(item) is not Incremental for item in typed.interventions)
        or tuple(typed.interventions) != declarations
        or tuple(estimator.incremental) != declarations
        or typed.reference != functional.reference
        or functional.outcome != study.design.outcome
        or functional.treatment != study.design.treatment
        or functional.adjustment != tuple(study.design.adjustment)
    ):
        raise CapabilityError(error)
    names = tuple(item.name for item in declarations)
    reference = names[0] if functional.reference is None else functional.reference
    if (
        len(set(names)) != len(names)
        or reference not in names
        or key.value not in names
        or key.reference != (reference if key.estimand == "ate_ipsi" else None)
        or (key.estimand == "ate_ipsi" and key.value == reference)
        or result.config.reference_arm != float(names.index(reference))
    ):
        raise CapabilityError(error)
    for nuisance in result.nuisances:
        state = nuisance.incremental
        mechanism = np.asarray(nuisance.propensity.values)
        if (
            type(state) is not IPSISet
            or mechanism.shape != (result.data.n, 2)
            or not np.all(np.isfinite(mechanism))
            or np.any((mechanism < 0.0) | (mechanism > 1.0))
            or not nuisance.propensity.simplex
            or not np.allclose(mechanism.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
            or nuisance.propensity.arms != result.data.arm_codes
            or nuisance.regimes is not None
            or nuisance.shifts is not None
            or nuisance.msm is not None
        ):
            raise CapabilityError(error)
        expected = IPSISet.evaluate(
            declarations, result.data, mechanism, reference=functional.reference
        )
        if (
            state.names != expected.names
            or state.deltas != expected.deltas
            or state.reference != expected.reference
            or any(
                not np.array_equal(getattr(state, slot), getattr(expected, slot))
                for slot in ("values", "weights", "derivative", "propensity")
            )
        ):
            raise CapabilityError(error)
    alias = parameter_name(key.estimand, arm=key.value, versus=key.reference)
    check_alias(result, estimand, key, alias, error)
    estimate = result.estimates[estimand]
    if estimate.scale != registered.scale or estimate.inference_value != estimate.psi:
        raise CapabilityError(error)
    if estimand in natural_incremental_means(result):
        raise CapabilityError(
            "simulated_confounding refuses a delta=1 incremental mean; it equals E[Y] "
            "and carries no counterfactual treatment dependence. Select a nontrivial "
            "incremental mean or a contrast against the natural course"
        )
    return estimator
