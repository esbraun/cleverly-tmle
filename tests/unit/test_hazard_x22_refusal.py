"""The targeting rules beside ``survival_model="hazard"`` (R-abc4).

The hazard walk of X14 moves one set of hazards along its own path in steps of ``step_size``.
It reads neither the iterate choice, nor a stopping rule, nor a step bound.  ``LTMLE`` refuses
those three settings beside it, before any data or learner, and the refusal names the roadmap
item that lifts it.  The defaults run the shipped walk untouched: the hazard solvers import no
function that the targeting options changed, and the fit is bit-identical with the settings written out.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from cleverly.exceptions import (
    NO_ALTERNATIVE,
    CapabilityError,
    MethodConfigurationError,
    RefusalReason,
)
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal import hazard as hazard_module
from cleverly.methods import TMLEMethod
from tests.discrete_law_point_survival import CellMeans, CellProbabilities, PointLaw

LAW = PointLaw("survival", 2)
REGIMENS = {"a0": 0, "a1": 1}

TAIL = " X33 in docs/roadmap.md tracks this. " + NO_ALTERNATIVE


def _message(names: str) -> str:
    return (
        "survival_model='hazard' walks its own path in steps of step_size, so "
        f"{names} do not apply to it." + TAIL
    )


class _Refuse(CellMeans):
    def fit(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("a learner ran")


def _refusing() -> dict[str, object]:
    return {
        "outcome_learner": _Refuse(),
        "treatment_learner": _Refuse(),
        "censoring_learner": _Refuse(),
    }


def _fit(**settings: Any) -> Any:
    return LTMLE(
        REGIMENS,
        n_folds=1,
        survival_model="hazard",
        step_size=2e-2,
        outcome_learner=CellMeans(),
        treatment_learner=CellProbabilities(),
        censoring_learner=CellMeans(),
        **settings,
    ).fit(LAW.frame(), **LAW.fit_columns())


CASES = [
    ({"iterate": "best"}, "iterate="),
    ({"stopping": "scaled_var"}, "stopping="),
    ({"stopping": "sample_size"}, "stopping="),
    ({"step_bound": 0.5}, "step_bound="),
    ({"iterate": "best", "stopping": "sample_size"}, "iterate= and stopping="),
    ({"stopping": "scaled_var", "step_bound": 0.5}, "stopping= and step_bound="),
    (
        {"iterate": "best", "stopping": "scaled_var", "step_bound": 0.5},
        "iterate=, stopping= and step_bound=",
    ),
]


@pytest.mark.parametrize(("settings", "names"), CASES, ids=[names for _, names in CASES])
@pytest.mark.parametrize("targeting", ["iterative", "one_step"])
def test_r_abc4_a_declared_rule_beside_the_hazard_walk_is_refused(
    settings: dict[str, Any], names: str, targeting: str
) -> None:
    with pytest.raises(CapabilityError, match=re.escape(_message(names))) as raised:
        LTMLE(
            REGIMENS,
            survival_model="hazard",
            targeting=targeting,
            **settings,
            **_refusing(),
        )
    refusal = raised.value.refusal
    assert refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert refusal.roadmap == "X33"
    assert refusal.alternative == NO_ALTERNATIVE


def test_r_abc4_the_refusal_precedes_the_data_and_every_learner() -> None:
    estimator = None
    with pytest.raises(CapabilityError):
        estimator = LTMLE(REGIMENS, survival_model="hazard", stopping="scaled_var", **_refusing())
    assert estimator is None


def test_r_abc4_a_malformed_rule_is_still_its_own_error() -> None:
    """Validity comes first: a bad value is the R-a1, R-b1 or R-c1 error, not the gap."""
    with pytest.raises(ValueError, match="stopping must be"):
        LTMLE(REGIMENS, survival_model="hazard", stopping="bogus", **_refusing())
    with pytest.raises(ValueError, match="iterate must be"):
        LTMLE(REGIMENS, survival_model="hazard", iterate="middle", **_refusing())
    with pytest.raises(ValueError, match="step_bound must be a positive finite number"):
        LTMLE(REGIMENS, survival_model="hazard", step_bound=-1.0, **_refusing())


@pytest.mark.parametrize(("settings", "names"), CASES[:4], ids=[names for _, names in CASES[:4]])
def test_r_abc4_the_iterated_mean_model_accepts_the_same_settings(
    settings: dict[str, Any], names: str
) -> None:
    """The control: the refusal is the hazard walk's, and the iterated-mean fit is unchanged."""
    LTMLE(REGIMENS, survival_model="iterated_mean", **settings, **_refusing())


def test_r_abc4_the_method_route_refuses_the_same_settings() -> None:
    method = TMLEMethod().with_overrides(survival_model="hazard", stopping="sample_size")
    with pytest.raises(CapabilityError, match=re.escape(_message("stopping="))):
        LTMLE(REGIMENS, **{**method.estimator_kwargs(longitudinal=True), **_refusing()})


def test_r_abc4_the_method_route_default_passes_the_defaults_through() -> None:
    kwargs = (
        TMLEMethod().with_overrides(survival_model="hazard").estimator_kwargs(longitudinal=True)
    )
    assert (kwargs["stopping"], kwargs["iterate"], kwargs["step_bound"]) == (
        "relative",
        "last",
        None,
    )
    LTMLE(REGIMENS, **{**kwargs, **_refusing()})


def test_the_hazard_walk_calls_no_function_x22_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hazard solvers import only ``relative_score`` and ``TargetingFailure`` of the targeting modules.

    Every fluctuation solver the targeting options changed raises when called, and a default hazard fit still
    runs, so the defaults reach the shipped path and nothing else.
    """
    from cleverly.estimators import targeting as targeting_module
    from cleverly.fluctuation import iterative as iterative_module
    from cleverly.longitudinal import sequential as sequential_module

    def tripwire(*args: object, **kwargs: object) -> object:
        raise AssertionError("a targeting solver ran on the hazard walk")

    for module, names in (
        (
            iterative_module,
            ("solve_fluctuation", "_solve_fluctuation_ruled", "_solve_fluctuation_bounded"),
        ),
        (targeting_module, ("solve_submodel",)),
        (sequential_module, ("solve_fluctuation", "_fluctuate_node")),
    ):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, tripwire)
    result = _fit()
    assert result.estimates
    assert {"relative_score", "TargetingFailure"} <= set(vars(hazard_module))


def test_the_written_out_defaults_are_bit_identical_to_the_omitted_ones() -> None:
    omitted = _fit()
    written = _fit(stopping="relative", iterate="last", step_bound=None, step_search=True)
    assert list(omitted.estimates) == list(written.estimates)
    for name in omitted.estimates:
        assert np.array_equal(omitted.estimates[name].psi, written.estimates[name].psi)
        assert np.array_equal(
            omitted.estimates[name].influence_curve, written.estimates[name].influence_curve
        )
        assert omitted.estimates[name].variance == written.estimates[name].variance


def test_the_method_error_class_is_not_used() -> None:
    """R-abc4 is a capability error, as the plan says, and not a configuration error."""
    with pytest.raises(CapabilityError) as raised:
        LTMLE(REGIMENS, survival_model="hazard", iterate="best", **_refusing())
    assert not isinstance(raised.value, MethodConfigurationError)
