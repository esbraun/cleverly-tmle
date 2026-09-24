"""Boundary behavior of the public omitted-variable reports."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cleverly.exceptions import CapabilityError
from cleverly.sensitivity import omitted_variable_bounds, robustness_value, sensitivity_elements
from tests.unit._exact_sensitivity_support import binary_oracle_fit


@pytest.fixture(scope="module")
def exact_fit() -> Any:
    return binary_oracle_fit()[0]


def test_zero_residual_variance_keeps_point_bounds_and_refuses_limits(exact_fit: Any) -> None:
    # Set each observed outcome to the targeted regression on the same fitted result.
    # The representer remains nonzero, so this isolates sigma2 = 0 from nu2 = 0.
    targeted = exact_fit.repeats[0].fluctuations["mean"].targeted.observed
    zero_fit = replace(exact_fit, data=replace(exact_fit.data, outcome=targeted))
    elements = sensitivity_elements(zero_fit)
    assert elements.sigma2 == 0.0
    assert elements.nu2 > 0.0
    assert elements.psi_max_bias is None
    bound = omitted_variable_bounds(zero_fit)
    assert bound.lower == bound.upper == exact_fit.psi("ate")
    assert bound.robustness_value is None
    assert bound._ci_lower is None and bound._ci_upper is None
    assert "limit_refusal" in bound.to_dict()
    assert "no equal strength reaches the null" in bound.summary()
    for name in (
        "ci_lower",
        "ci_upper",
        "robustness_value_ci",
        "plugin_interval_lower",
        "plugin_interval_upper",
        "robustness_value_plugin_interval",
    ):
        with pytest.raises(CapabilityError, match="max_bias is zero"):
            getattr(bound, name)
    report = robustness_value(zero_fit)
    assert report["rv"] is None
    assert "rva" not in report
    assert "max_bias is zero" in report["limit_refusal"]
    assessment = zero_fit.sensitivity.run_all(arguments={"robustness_value": {"estimand": "ate"}})[
        "robustness_value"
    ]
    assert "point robustness value unavailable" in assessment.detail
    assert "max_bias is zero" in assessment.detail

    # Nonzero witness: the same public operation on the original fit has a curve.
    positive = omitted_variable_bounds(exact_fit)
    assert np.isfinite(positive.ci_lower)
    assert positive.robustness_value is not None


def test_zero_alignment_cannot_reach_an_outside_null(exact_fit: Any) -> None:
    bound = omitted_variable_bounds(exact_fit, rho=0.0)
    assert bound.lower == bound.upper == exact_fit.psi("ate")
    assert bound.robustness_value is None
    assert bound.robustness_value_ci is None
    assert bound.to_dict()["robustness_value"] is None
    assert bound.to_dict()["robustness_value_ci"] is None
    assert "unavailable: no equal strength reaches the null" in bound.summary()
    report = robustness_value(exact_fit, rho=0.0)
    assert report["rv"] is None and report["rva"] is None
    assessment = exact_fit.sensitivity.run_all(arguments={"robustness_value": {"rho": 0.0}})[
        "robustness_value"
    ]
    assert assessment.detail.count("unavailable: no equal strength reaches the null") == 2


def test_a_baseline_limit_crossing_the_null_has_zero_rva(exact_fit: Any) -> None:
    null = exact_fit.psi("ate") - 0.01
    report = robustness_value(exact_fit, null_hypothesis=null)
    assert report["rv"] is not None and report["rv"] > 0.0
    assert report["rva"] == 0.0
    assessment = exact_fit.sensitivity.run_all(
        arguments={"robustness_value": {"null_hypothesis": null}}
    )["robustness_value"]
    assert assessment.detail.endswith("confidence-limit value 0")
    assert robustness_value(exact_fit, null_hypothesis=exact_fit.psi("ate"))["rv"] == 0.0

    # The positive control solves the non-crossing limit equation, not a nearby minimum.
    outside = omitted_variable_bounds(exact_fit)
    assert outside.robustness_value_ci is not None
    at_threshold = omitted_variable_bounds(
        exact_fit,
        cf_y=outside.robustness_value_ci,
        cf_d=outside.robustness_value_ci,
    )
    assert at_threshold.ci_lower == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("level", [0.0, 0.25, 1.0, np.nan])
def test_an_invalid_one_sided_level_is_refused(exact_fit: Any, level: float) -> None:
    with pytest.raises(ValueError, match="level must lie in"):
        omitted_variable_bounds(exact_fit, level=level)
    with pytest.raises(ValueError, match="level must lie in"):
        robustness_value(exact_fit, level=level)
