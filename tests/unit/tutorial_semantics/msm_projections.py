"""The reviewed semantic callback for ``docs/examples/msm-projections``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def check(namespace: dict[str, Any]) -> None:
    """The arm-mean question names its actual support instead of a binary surrogate."""
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

    population = namespace["population"]
    assert population[0] < population[1] < population[2]
    assert population[2] - population[1] > population[1] - population[0]
    # The page refuses text cadence labels. Only the MSM.linear refusal may satisfy it.
    assert "reads the treatment level" in namespace["refusal"]

    projection = namespace["projection"]
    trend = namespace["trend_result"]
    for name, target in zip(
        ("msm[(intercept)]", "msm[assigned contacts]"), projection, strict=True
    ):
        assert trend[name].ci[0] <= target <= trend[name].ci[1]

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
