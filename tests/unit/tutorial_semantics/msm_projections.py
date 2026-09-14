"""The reviewed semantic callback for ``docs/examples/msm-projections``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.datasets import navigation_protocol
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
)

NOTEBOOK = EXAMPLES / "msm-projections.ipynb"


def check(namespace: dict[str, Any]) -> None:
    """The projection tutorial's protocol, failure mode, control, and refusal claims hold."""
    effect = namespace["arms"]
    fitted = namespace["arm_result"]
    levels = tuple(namespace["study"].data.treatment_levels)
    positivity = " ".join(
        assumption
        for assumption in effect.identification.assumptions
        if "positivity" in assumption.lower()
    )
    summary = effect.summary()

    assert len(levels) == 3, "the tutorial no longer witnesses multi-arm identification"
    assert fitted.nuisance.propensity.values.shape[1] == len(levels)
    for level in levels:
        assert str(level) in positivity
    assert "treatment_mechanism" in summary
    assert "P(A = 1 | W)" not in summary
    assert "both counterfactual means" not in summary

    # The protocol step prints the record, and both fits carry its digest.
    # The reading names the fields this page changes in the program protocol.
    assert changed_fields(namespace["protocol"], navigation_protocol()) == {
        "time_zero",
        "treatment_strategies",
        "treatment_versions",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    assert_protocol_recorded(
        NOTEBOOK, "protocol", namespace["protocol"], fitted, namespace["trend_result"]
    )

    population = namespace["population"]
    assert population[0] < population[1] < population[2]
    assert population[2] - population[1] > population[1] - population[0]
    # "The observed means put medium above high", while the true means reverse that order.
    observed = namespace["observed"]
    assert observed.loc["medium", "transition_score"] > observed.loc["high", "transition_score"]
    assert observed.loc["medium", "discharge_risk"] > observed.loc["high", "discharge_risk"]
    # "Each interval contains its true mean on this draw."
    for arm, target in zip(namespace["ARMS"], population, strict=True):
        assert covers(fitted[f"ey[{arm}]"], target)
    # The page refuses text cadence labels. Only the MSM.linear refusal may satisfy it.
    assert "reads the treatment level" in namespace["refusal"]

    projection = namespace["projection"]
    trend = namespace["trend_result"]
    for name, target in zip(
        ("msm[(intercept)]", "msm[assigned contacts]"), projection, strict=True
    ):
        assert covers(trend[name], target)
    # Under the uniform weight the slope is the fixed contrast of the three means, exactly.
    assert namespace["contrast_slope"] == pytest.approx(projection[1], rel=1e-12, abs=1e-12)
    # The share weight moves the population slope: the weight is part of the estimand.
    assert abs(namespace["share_projection"][1] - projection[1]) > 0.005

    saturated = namespace["saturated_result"]
    mappings = {
        "msm[(intercept)]": namespace["arm_result"]["ey[low]"],
        "msm[medium vs low]": namespace["arm_contrasts"]["ate[medium vs low]"],
        "msm[high vs low]": namespace["arm_contrasts"]["ate[high vs low]"],
    }
    for name, counterpart in mappings.items():
        coefficient = saturated[name]
        assert coefficient.psi == pytest.approx(counterpart.psi, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(
            coefficient.influence_curve,
            counterpart.influence_curve,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(coefficient.ci, counterpart.ci, rtol=1e-12, atol=1e-12)

    line = namespace["design"] @ projection
    residual = line - population
    assert abs(residual[1]) > 0.15, "the contact mapping no longer witnesses misspecification"
    assert 0.17 < abs(residual[1]) < 0.21, "the prose says the line misses medium by about 0.19"
    # "Well below" the medium interval: the gap is measured at about 3.5 standard errors.
    medium = namespace["arm_result"]["ey[medium]"]
    gap = medium.ci[0] - namespace["estimated_line"][1]
    assert gap > 2.0 * medium.std_error, (
        "on this draw the estimated line no longer falls well below the medium interval"
    )

    # The assessment names the support warning, and the slope barely moves along the curve
    # even where the largest bound clips most of the rows.
    assessment = namespace["assessment"]
    assert "support" in {item.name for item in assessment.attention}
    slope_curve = namespace["slope_curve"]
    assert slope_curve["truncated_fraction"].iloc[-1] > 0.5
    assert slope_curve["delta_from_fitted"].abs().max() < 0.005

    # No omitted-variable bound covers an MSM coefficient; the arm contrasts do have one.
    assert "Riesz representer" in namespace["sensitivity_refusal"]
    robustness = namespace["robustness"]
    assert 0.0 < robustness["ate[medium vs low]"]["rv"] < robustness["ate[high vs low]"]["rv"]
