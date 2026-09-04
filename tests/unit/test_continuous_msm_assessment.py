"""Continuous MSM assessment retains the unavailable support diagnostic."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from cleverly import AssessmentStatus
from cleverly.assessment import DiagnosticsFacade, assessment_capabilities
from cleverly.exceptions import CapabilityError, DataError
from tests.unit.test_simulated_confounding_msm import _GRID, _fit_continuous
from tests.unit.test_simulated_confounding_policies import _fit_msm


@pytest.mark.parametrize("link", ["identity", "log", "logit"])
def test_continuous_msm_capability_refuses_the_arm_support_fallback(
    link: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fit_continuous(link)
    from cleverly.sensitivity import positivity

    monkeypatch.setattr(
        positivity, "positivity_report", lambda *_: pytest.fail("used arm support for doses")
    )
    declared = next(row for row in assessment_capabilities(result) if row.operation == "support")
    capability = result.diagnostics.capability("support")
    assert not declared.available and not capability.available
    assert capability.status is AssessmentStatus.UNAVAILABLE
    assert "dose-grid support diagnostic" in capability.reason
    with pytest.raises(CapabilityError, match="dose-grid support diagnostic"):
        result.diagnostics.support()
    validation = result.validate()
    assert validation["support"].status is AssessmentStatus.UNAVAILABLE
    assert capability.reason == validation["support"].detail
    assert validation["score_equations"].status is not AssessmentStatus.UNAVAILABLE
    assert validation["nuisance_models"].status is not AssessmentStatus.UNAVAILABLE
    assert [model.name for model in result.diagnostics.nuisance_models().models] == ["outcome"]
    assert result.diagnostics.run_all()["support"].status is AssessmentStatus.UNAVAILABLE


def test_full_continuous_msm_assessment_reaches_the_requested_surface() -> None:
    result = _fit_continuous()
    kwargs = {"estimand": "msm[a]", "grid": _GRID, "random_state": 31}
    expected = result.sensitivity.simulated_confounding(**kwargs)
    report = result.assess(include_refits=True, arguments={"simulated_confounding": kwargs})
    assert report.validation["support"].status is AssessmentStatus.UNAVAILABLE
    assert report.diagnostics["support"].status is AssessmentStatus.UNAVAILABLE
    item = report.sensitivity["simulated_confounding"]
    assert item.status is AssessmentStatus.COMPLETED
    assert item._report is expected


def test_binary_msm_keeps_its_available_arm_support() -> None:
    result = _fit_msm()
    assert result.diagnostics.capability("support").available
    assert result.diagnostics.support().n == result.data.n


def test_validation_does_not_swallow_errors_from_available_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Use a fresh result so the deliberate failure cannot be hidden by a cached report.
    result = replace(_fit_continuous())

    def broken(_: Any) -> Any:
        raise DataError("deliberate score diagnostic defect")

    monkeypatch.setattr(DiagnosticsFacade, "score_equations", broken)
    with pytest.raises(DataError, match="deliberate score diagnostic defect"):
        result.validate()
