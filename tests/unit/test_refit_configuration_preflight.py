"""A replay slot reads every guard that a nuisance refit runs before learning."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest

from cleverly.assessment import POINT_REPLAY_REFIT_CONFIGURATION, replayability
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from tests.unit._capability_sweep_support import fit_rr_missing


@pytest.fixture(scope="module")
def rr_result() -> Any:
    return fit_rr_missing()


def _never_fit(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("a learner was reached before the configuration refusal")


def test_replay_reads_cv_parameter_guard_before_any_learner(
    rr_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    estimator = copy.copy(rr_result.estimator)
    estimator.cv_evaluation = True
    result = replace(rr_result, estimator=estimator)
    monkeypatch.setattr(estimator, "_nuisances", _never_fit)

    reason = estimator._refit_configuration_refusal(result.data)
    assert reason is not None
    assert "cv_evaluation=True does not yet support ['rr']" in reason
    replay = replayability(result)
    assert not replay.refit_nuisances
    assert replay.unreconstructible == (POINT_REPLAY_REFIT_CONFIGURATION,)
    with pytest.raises(ValueError) as raised:
        estimator.refit(result.data)
    assert str(raised.value) == reason


@pytest.mark.parametrize(
    ("setting", "value", "fragment"),
    [
        ("q_bounds", (0.0, 1.0), "q_bounds does not apply to a binary outcome"),
        ("g_bounds", (0.1, 1.0), "g_bounds pair must satisfy"),
        ("reference", "unknown-arm", "reference='unknown-arm' is not a level"),
    ],
)
def test_replay_reads_scaler_and_config_guards_before_any_learner(
    rr_result: Any,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: Any,
    fragment: str,
) -> None:
    estimator = copy.copy(rr_result.estimator)
    setattr(estimator, setting, value)
    result = replace(rr_result, estimator=estimator)
    monkeypatch.setattr(estimator, "_nuisances", _never_fit)

    reason = estimator._refit_configuration_refusal(result.data)
    assert reason is not None
    assert fragment in reason
    replay = replayability(result)
    assert not replay.refit_nuisances
    assert replay.unreconstructible == (POINT_REPLAY_REFIT_CONFIGURATION,)
    with pytest.raises(ValueError) as raised:
        estimator.refit(result.data)
    assert str(raised.value) == reason


def test_preflight_keeps_subclass_estimand_override(
    rr_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RefusingTMLE(TMLE):
        def _resolve_estimands_for_data(self, data: CausalData) -> tuple[str, ...]:
            raise ValueError("subclass preflight refusal")

    estimator = copy.copy(rr_result.estimator)
    estimator.__class__ = RefusingTMLE
    result = replace(rr_result, estimator=estimator)
    monkeypatch.setattr(estimator, "_nuisances", _never_fit)

    assert estimator._refit_configuration_refusal(result.data) == "subclass preflight refusal"
    assert not replayability(result).refit_nuisances
    with pytest.raises(ValueError, match="subclass preflight refusal"):
        estimator.refit(result.data)
