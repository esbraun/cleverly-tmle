"""Validate and freeze baseline policies for complete simulated-confounding refits.

The surface changes A and Y only. A known policy, or an arm-based MSM design, therefore
keeps its evaluated baseline arrays. Validate the original declaration against every
stored draw before copying those arrays into a replay estimator. No targeting or
nuisance calculation is replaced here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..data import CausalData
from ..exceptions import CapabilityError, DataError
from ..interventions import RegimeSet, Rule, Static, Stochastic
from ..interventions.base import _SIMPLEX_TOLERANCE, as_interventions
from ..msm import MSM, MSMSet, _DataBoundArmFunction
from ..provenance import fingerprint_array
from ..study import MSMProjection, PointTreatment, RegimeContrast, RegimeMean
from ..targets import TARGETS
from ..targets.base import parameter_name
from ..utils.frames import as_frame, matrix_from_columns

_FIXED_TARGET_TYPES: dict[str, type] = {
    "ey_regime": RegimeMean,
    "ate_regime": RegimeContrast,
    "msm": MSMProjection,
}


def fixed_axis(target: str) -> str | None:
    """Return the parameter axis of an audited fixed-policy target."""
    if target not in _FIXED_TARGET_TYPES:
        return None
    return "msm" if target == "msm" else "regime"


def fixed_replay_refusal(estimator: Any, target: str) -> str | None:
    """Describe a fixed-policy boundary without evaluating user callbacks."""
    from ..estimators.tmle import TMLE

    if fixed_axis(target) is None:
        return None
    if type(estimator) is not TMLE:
        return (
            "simulated_confounding supports fixed regimes and MSMs under exact ordinary TMLE only"
        )
    if target == "msm":
        model = estimator.msm
        if type(model) is not MSM or model.link != "identity" or model.doses:
            return "simulated_confounding supports an identity-link arm-based MSM only"
    elif any(type(item) not in {Static, Rule, Stochastic} for item in estimator.interventions):
        return "simulated_confounding supports exact Static, Rule, and Stochastic regimes only"
    return None


@dataclass(frozen=True)
class _BaselineRows:
    names: tuple[str, ...]
    fingerprint: str
    levels: tuple[Any, ...]

    @classmethod
    def from_data(cls, data: CausalData) -> _BaselineRows:
        return cls(data.covariate_names, fingerprint_array(data.covariates), data.treatment_levels)

    def check_data(self, data: CausalData) -> None:
        if (
            data.covariate_names != self.names
            or data.treatment_levels != self.levels
            or fingerprint_array(data.covariates) != self.fingerprint
        ):
            raise CapabilityError(
                "fixed-policy replay requires the original baseline rows and arms"
            )

    def check_frame(self, frame: Any) -> None:
        values = matrix_from_columns(as_frame(frame), self.names)
        if fingerprint_array(values) != self.fingerprint:
            raise CapabilityError("fixed-policy replay requires the original baseline rows")


@dataclass(frozen=True)
class _FrozenRegime:
    name: str
    values: Any
    baseline: _BaselineRows

    def density(self, data: CausalData) -> Any:
        self.baseline.check_data(data)
        return self.values.copy()


@dataclass(frozen=True)
class _FrozenArmFunction(_DataBoundArmFunction):
    values: Any
    baseline: _BaselineRows

    def check_data(self, data: CausalData) -> None:
        self.baseline.check_data(data)

    def __call__(self, arm: Any, frame: Any) -> Any:
        self.baseline.check_frame(frame)
        try:
            index = self.baseline.levels.index(arm)
        except ValueError as cause:
            raise CapabilityError(
                "fixed MSM replay requires the original treatment arms"
            ) from cause
        return self.values[:, index].copy()


def _same_items(left: Any, right: Any) -> bool:
    """Compare policy declarations without invoking arbitrary callback equality."""
    left_items, right_items = tuple(left), tuple(right)
    return len(left_items) == len(right_items) and all(
        a is b for a, b in zip(left_items, right_items, strict=True)
    )


def _same_arrays(left: Any, right: Any, names: tuple[str, ...]) -> bool:
    return all(np.array_equal(getattr(left, name), getattr(right, name)) for name in names)


def _checked_regimes(state: Any, data: CausalData) -> RegimeSet:
    if type(state) is not RegimeSet:
        raise DataError("the fit lost its cached regime densities")
    checked = RegimeSet(state.names, state.values, state.reference)
    values = checked.values
    if (
        values.shape != (data.n, data.n_arms, len(checked.names))
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(np.abs(values.sum(axis=1) - 1.0) > _SIMPLEX_TOLERANCE)
    ):
        raise DataError(
            "cached regime densities must be finite probability simplexes on fitted rows"
        )
    return checked


def _freeze_regimes(result: Any, key: Any, typed: Any, functional: Any) -> tuple[Any, str]:
    data, estimator = result.data, result.estimator
    declarations = as_interventions(functional.interventions)
    if (
        not declarations
        or not _same_items(typed.regimens, functional.interventions)
        or len(estimator.interventions) != len(declarations)
        or any(
            not (a is b or (type(a) is Static and type(b) is Static and a == b))
            for a, b in zip(estimator.interventions, declarations, strict=True)
        )
        or any(type(item) not in {Static, Rule, Stochastic} for item in declarations)
        or typed.reference != functional.reference
        or typed.horizons is not None
        or functional.msm is not None
        or estimator.msm is not None
        or key.term is not None
    ):
        raise DataError("regime declarations disagree")
    expected = _checked_regimes(
        RegimeSet.evaluate(declarations, data, reference=functional.reference), data
    )
    for nuisance in result.nuisances:
        state = _checked_regimes(nuisance.regimes, data)
        if (
            state.names != expected.names
            or state.reference != expected.reference
            or not np.array_equal(state.values, expected.values)
            or nuisance.msm is not None
        ):
            raise DataError("declared regime densities disagree with a stored cross-fitting draw")
    reference = expected.names[int(expected.reference)]
    if (
        result.config.reference_arm != expected.reference
        or key.value not in expected.names
        or key.reference != (reference if key.estimand == "ate_regime" else None)
        or (key.estimand == "ate_regime" and key.value == reference)
    ):
        raise DataError("regime labels or reference disagree")
    alias = parameter_name(key.estimand, arm=key.value, versus=key.reference)
    replay = copy.copy(estimator)
    baseline = _BaselineRows.from_data(data)
    replay.interventions = tuple(
        _FrozenRegime(name, expected.values[:, :, index].copy(), baseline)
        for index, name in enumerate(expected.names)
    )
    return replay, alias


def _freeze_msm(result: Any, key: Any, typed: Any, functional: Any) -> tuple[Any, str]:
    data, estimator = result.data, result.estimator
    model = functional.msm
    if (
        type(model) is not MSM
        or typed.model is not model
        or estimator.msm is not model
        or model.link != "identity"
        or model.doses
        or typed.regimens is not None
        or typed.horizons is not None
        or functional.interventions
        or estimator.interventions
        or functional.reference is not None
        or key.value is not None
        or key.reference is not None
        or key.term not in model.terms
    ):
        raise DataError("identity-link arm-based MSM declarations disagree")
    expected = MSMSet.evaluate(model, data)
    for nuisance in result.nuisances:
        state = nuisance.msm
        if type(state) is not MSMSet:
            raise DataError("the fit lost its cached MSM design")
        # Re-run the array validation and rank guard without evaluating user functions.
        checked = replace(state)
        if (
            checked.terms != expected.terms
            or checked.arms != expected.arms
            or checked.link != expected.link
            or checked.dose_values
            or not _same_arrays(
                checked,
                expected,
                ("design", "weights", "clever_weights", "observed_design", "observed_weights"),
            )
            or nuisance.regimes is not None
        ):
            raise DataError("declared MSM arrays disagree with a stored cross-fitting draw")
    replay = copy.copy(estimator)
    baseline = _BaselineRows.from_data(data)
    replay.msm = replace(
        model,
        design=_FrozenArmFunction(expected.design.copy(), baseline),
        weights=_FrozenArmFunction(expected.weights.copy(), baseline),
    )
    return replay, parameter_name("msm", arm=key.term)


def validate_fixed_replay(result: Any, estimand: str, key: Any) -> Any:
    """Return a copied estimator with validated fixed baseline-policy arrays."""
    from ._simulated_confounding_request import _population_alias

    identified, estimator = result.identified_effect, result.estimator
    functional, typed = identified.functional, identified.estimand
    target = key.estimand
    axis = fixed_axis(target)
    registered = TARGETS.get(target)
    error = "simulated_confounding found inconsistent fixed-policy parameter metadata"
    study = getattr(identified, "_study", None)
    if (
        axis is None
        or type(typed) is not _FIXED_TARGET_TYPES[target]
        or functional.target != target
        or functional.axis != axis
        or key.axis != axis
        or result.config.parameter_axis != axis
        or target not in result.config.estimands
        or target not in tuple(estimator.estimands or ())
        or estimator.reference != functional.reference
        or estimator.shifts
        or estimator.incremental
        or functional.longitudinal
        or functional.horizons is not None
        or functional.intermediate is not None
        or registered is None
        or registered.parameter_axis != axis
        or identified.identification != registered.identification
        or key.regimen is not None
        or key.cause is not None
        or key.horizon is not None
        or study is None
        or type(study.design) is not PointTreatment
        or functional.outcome != study.design.outcome
        or functional.treatment != study.design.treatment
        or functional.adjustment != tuple(study.design.adjustment)
    ):
        raise CapabilityError(error)
    try:
        study.design._check_prepared(result.data)
        replay, alias = (
            _freeze_msm(result, key, typed, functional)
            if axis == "msm"
            else _freeze_regimes(result, key, typed, functional)
        )
    except DataError as cause:
        raise CapabilityError(f"{error}: {cause}") from cause
    estimate = result.estimates[estimand]
    if (
        key.alias != estimand
        or _population_alias(result, key, alias) != estimand
        or estimate.name != estimand
        or estimate.scale != registered.scale
        or not np.isfinite(estimate.psi)
        or estimate.inference_value != estimate.psi
    ):
        raise CapabilityError(error)
    return replay
