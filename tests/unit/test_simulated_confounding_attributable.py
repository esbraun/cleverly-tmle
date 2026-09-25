"""Population attributable surfaces: complete replay and observed-law witnesses."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cleverly import (
    AssessmentStatus,
    NaturalCourseMean,
)
from cleverly.estimators import CTMLE, DRTMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity._simulated_confounding_request import (
    _BINARY_PARAMETER_TARGETS,
    _eligible_binary_parameter_names,
)
from cleverly.targets import TARGETS
from cleverly.targets.base import parameter_name, stratum_alias
from tests.unit._confounding_support import (
    alias_for as _alias,
)
from tests.unit._confounding_support import (
    baseline_mask as _mask,
)
from tests.unit._confounding_support import (
    forbid_draw_and_refit as _forbid_draw_and_refit,
)
from tests.unit._confounding_support import (
    replacement as _replacement,
)
from tests.unit._simulated_confounding_support import _ZERO_CELL_STRENGTH, _fit_attributable


@pytest.mark.parametrize(
    "target,family", [("par", "gaussian"), ("par", "binomial"), ("paf", "binomial")]
)
@pytest.mark.parametrize("stratum", [None, ("small",)])
def test_attributable_cells_equal_complete_weighted_refits(
    target: str,
    family: str,
    stratum: tuple[str, ...] | None,
) -> None:
    result = _fit_attributable(target, family)
    alias = _alias(result, target, stratum)
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0, 0.17)),
        random_state=31,
    )
    replacement = _replacement(result, surface, 0.22, 0.17)
    manual = result.estimator.refit(replacement, random_state=31)
    cell = surface.cells[-1]
    assert surface.complete
    assert cell.estimate == manual[alias].psi
    assert cell.displacement == manual[alias].psi - result[alias].psi
    assert abs(cell.displacement) > 1e-5
    assert surface.movement_scale == "estimate_difference"
    assert surface.population == "baseline"
    assert surface.stratum == stratum
    assert cell.target_population_fraction == 1.0
    assert surface.target_measure == "fixed_empirical_tilt"
    np.testing.assert_array_equal(replacement.weights, result.data.weights)
    np.testing.assert_array_equal(replacement.strata, result.data.strata)
    assert replacement.n == result.data.n


@pytest.mark.parametrize(
    "target,family", [("par", "gaussian"), ("par", "binomial"), ("paf", "binomial")]
)
@pytest.mark.parametrize("reference", [None, 0, 1, "control", "active"])
def test_observed_and_reference_components_move_with_the_cell(
    target: str,
    family: str,
    reference: Any,
) -> None:
    """Separate the published composition from each tempting incorrect composition.

    Independent counterfactual-mean refits supply both intervention means. Empirical
    outcome averages supply the observed-law term, without reading PAR/PAF internals.

    ``reference=None`` is the default-constructed form the study API suggests, and
    ``_validate_binary_parameter_state`` resolves it to the first treatment level. No
    explicit case reaches that fallback, so this row is its witness: read the label back
    from the fit rather than restate it here, and a fallback that resolved a different
    arm refuses the surface instead of moving the wrong mean.
    """
    result = _fit_attributable(target, family, reference=reference)
    declared = result.data.arm_label(result.config.reference_arm)
    if reference is None:
        assert declared == result.data.treatment_levels[0]
    else:
        assert declared == reference
    alias = _alias(result, target, ("small",))
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0, 0.27)),
        random_state=31,
    )
    changed = _replacement(result, surface, 0.22, 0.27)
    means_estimator = copy(result.estimator)
    means_estimator.estimands = ("ey",)
    means = means_estimator.refit(changed, random_state=31)
    mask = _mask(result, ("small",))
    observed = np.average(changed.outcome[mask], weights=changed.weights[mask])
    frozen = np.average(result.data.outcome[mask], weights=result.data.weights[mask])
    by_arm = {
        arm: means[
            stratum_alias(
                parameter_name("ey", arm=arm),
                changed.stratum_label(changed.strata_levels.index(("small",))),
            )
        ].psi
        for arm in result.data.treatment_levels
    }
    intervention = by_arm[declared]
    wrong_arm = next(value for arm, value in by_arm.items() if arm != declared)
    expected = observed - intervention if target == "par" else 1 - intervention / observed
    wrong = (
        [frozen - intervention, intervention - observed, observed - wrong_arm]
        if target == "par"
        else [
            1 - intervention / frozen,
            intervention / observed,
            1 - observed / intervention,
            1 - wrong_arm / observed,
        ]
    )
    assert abs(observed - frozen) > 1e-3
    assert surface.cells[-1].estimate == pytest.approx(expected, abs=1e-12)
    assert all(abs(expected - alternative) > 1e-3 for alternative in wrong)
    assert result.parameter_keys[alias].value == declared
    assert result.parameter_keys[alias].reference is None


def test_negative_fraction_uses_identity_movement_and_repeat_aggregation() -> None:
    result = _fit_attributable("paf", reference=1, repeats=3)
    alias = _alias(result, "paf", ("small",))
    kwargs = {
        "estimand": alias,
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0,)),
        "random_state": 31,
    }
    surface = result.sensitivity.simulated_confounding(**kwargs)
    manual = result.estimator.refit(_replacement(result, surface, 0.22, 0.0), random_state=31)
    assert surface.complete
    assert result[alias].psi < 0
    assert surface.cells[-1].estimate < 0
    assert surface.cells[-1].estimate == np.median([draw.psi[alias] for draw in manual.repeats])
    assert abs(surface.cells[-1].estimate - manual.repeats[0].psi[alias]) > 1e-3
    assert surface.cells[-1].displacement == manual[alias].psi - result[alias].psi
    assert surface.n_repeats == 3
    assert result.sensitivity.simulated_confounding(**kwargs) is surface
    restored = loads(dumps(result))
    assert restored.sensitivity.simulated_confounding(**kwargs) == surface
    battery = result.assess(include_refits=True, arguments={"simulated_confounding": kwargs})
    item = battery.sensitivity["simulated_confounding"]
    assert item.status is AssessmentStatus.COMPLETED
    assert item._report is surface


def test_zero_observed_risk_retains_a_failed_fraction_cell() -> None:
    result = _fit_attributable("paf", strata=False, zero_cell=True)
    surface = simulated_confounding(
        result,
        "paf",
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0, _ZERO_CELL_STRENGTH)),
        random_state=31,
    )
    changed = _replacement(result, surface, 0.0, _ZERO_CELL_STRENGTH)
    assert np.count_nonzero(changed.outcome) == 0
    assert surface.cells[0].failure is None
    failed = surface.cells[1]
    assert failed.failure is not None
    assert "zero" in failed.failure.message
    assert failed.estimate is None and failed.displacement is None
    assert failed.outcome_strength == _ZERO_CELL_STRENGTH
    assert failed.induced_treatment_association is not None
    assert not surface.complete


@pytest.mark.parametrize("target", ["par", "paf"])
@pytest.mark.parametrize(
    "change", ["typed", "functional", "key-value", "key-reference", "alias", "replay", "reference"]
)
def test_attributable_metadata_refuses_before_randomness(
    target: str,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit_attributable(target, strata=False)
    identified = result.identified_effect
    if change == "typed":
        result = replace(
            result, identified_effect=replace(identified, estimand=NaturalCourseMean())
        )
    elif change == "functional":
        result = replace(
            result,
            identified_effect=replace(
                identified, functional=replace(identified.functional, reference=1)
            ),
        )
    elif change.startswith("key") or change == "alias":
        key = result.parameter_keys[target]
        field = {"key-value": "value", "key-reference": "reference", "alias": "alias"}[change]
        result = replace(
            result,
            parameter_keys={target: replace(key, **{field: "wrong" if field == "alias" else 1})},
        )
    else:
        estimator = copy(result.estimator)
        if change == "replay":
            estimator.estimands = ("ey_obs",)
        else:
            estimator.reference = 1
        result = replace(result, estimator=estimator)
    _forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match=r"inconsistent.*metadata"):
        simulated_confounding(
            result, target, grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
        )


@pytest.mark.parametrize("target", ["par", "paf"])
@pytest.mark.parametrize("method", ["greedy", "ordered", "discrete", "oat", "drtmle"])
def test_unevidenced_attributable_fit_refuses_upstream(target: str, method: str) -> None:
    with pytest.raises(CapabilityError, match=r"no .* evidenced for this functional"):
        _fit_attributable(target, method=method, strata=False)


@pytest.mark.parametrize("target", ["par", "paf"])
def test_unreplayable_attributable_estimators_are_withheld(
    target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit_attributable(target)
    alias = _alias(result, target, ("small",))
    for estimator in [CTMLE(strategy="greedy"), CTMLE(strategy="oat"), DRTMLE()]:
        forged = replace(result, estimator=estimator)
        _forbid_draw_and_refit(monkeypatch, estimator)
        assert alias not in _eligible_binary_parameter_names(forged)
        with pytest.raises(CapabilityError, match="PAR and PAF"):
            simulated_confounding(
                forged, alias, grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
            )


def test_natural_course_refuses_and_is_absent_from_suggestions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit_attributable("ey_obs", strata=False)
    _forbid_draw_and_refit(monkeypatch, result.estimator)
    assert _eligible_binary_parameter_names(result) == ()
    with pytest.raises(CapabilityError, match=r"natural|counterfactual"):
        result.sensitivity.simulated_confounding(
            "ey_obs", grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
        )
    with pytest.raises(CapabilityError, match=r"natural|counterfactual"):
        result.sensitivity.simulated_confounding(
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
        )
    with pytest.raises(ValueError, match="unavailable") as error:
        simulated_confounding(
            result, "unknown", grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
        )
    assert "'ey_obs'" not in str(error.value)


@pytest.mark.parametrize("target", ["par", "paf"])
def test_attributable_backend_parity_and_sole_alias_facade(target: str) -> None:
    kwargs = {
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0,)),
        "random_state": 31,
    }
    left = _fit_attributable(target, strata=False).sensitivity.simulated_confounding(**kwargs)
    right = _fit_attributable(
        target, strata=False, backend="polars"
    ).sensitivity.simulated_confounding(**kwargs)
    assert left.estimand == right.estimand == target
    for a, b in zip(left.cells, right.cells, strict=True):
        assert a.estimate == pytest.approx(b.estimate, abs=1e-12)
        assert a.displacement == pytest.approx(b.displacement, abs=1e-12)


def test_registered_point_targets_have_an_explicit_surface_disposition() -> None:
    """Every registered point target carries a written disposition, not a bare membership.

    The point of the check is the sentence beside each name. A group whose members are a
    bare set records that somebody sorted the target, and not why the surface treats it the
    way it does. ``policy`` and ``refused`` therefore both map a name to its reason.
    """
    binary = {"ate", "att", "atc", "ey", "ey1", "ey0", "par", "paf", "rr", "or"}
    continuous = {"ey_shift", "ate_shift"}
    policy = {
        "ey_regime": "fixed-regime mean, replayed on the regime axis under ordinary TMLE",
        "ate_regime": "fixed-regime contrast, replayed on the regime axis under ordinary TMLE",
        "ey_ipsi": "incremental mean, replayed on the ipsi axis for a marginal request",
        "ate_ipsi": "incremental contrast, replayed on the ipsi axis for a marginal request",
        "msm": "working-model coefficient, replayed on the msm axis under ordinary TMLE",
    }
    refused = {
        "ey_obs": "natural-course mean has no counterfactual treatment term",
    }
    assert binary == set(_BINARY_PARAMETER_TARGETS)
    dispositions = (binary, continuous, policy.keys(), refused.keys())
    assert all(policy.values()) and all(refused.values())
    assert sum(len(set(group)) for group in dispositions) == len(set().union(*dispositions))
    assert set().union(*dispositions) == TARGETS.keys()


@pytest.mark.parametrize("target", ["par", "paf"])
@pytest.mark.parametrize(
    "change",
    ["scale", "name", "nonfinite", "config", "registry-scale", "registry-family", "registry-axis"],
)
def test_attributable_identity_and_registry_metadata_refuse_before_draws(
    target: str,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit_attributable(target, strata=False)
    if change.startswith("registry-"):
        field, value = {
            "registry-scale": ("scale", "ratio"),
            "registry-family": ("requires_family", "gaussian"),
            "registry-axis": ("parameter_axis", "shift"),
        }[change]
        monkeypatch.setitem(TARGETS, target, replace(TARGETS[target], **{field: value}))
    elif change == "config":
        result = replace(result, config=replace(result.config, reference_arm=1.0))
    else:
        field, value = {
            "scale": ("scale", "ratio"),
            "name": ("name", "wrong"),
            "nonfinite": ("psi", float("nan")),
        }[change]
        result = replace(result, estimates={target: replace(result[target], **{field: value})})
    _forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match=r"inconsistent.*metadata"):
        simulated_confounding(
            result, target, grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
        )


def test_fraction_family_refuses_before_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _fit_attributable("paf", strata=False)
    result = replace(result, data=replace(result.data, family="gaussian"))
    _forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match="binary outcome only"):
        simulated_confounding(
            result, "paf", grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
        )
