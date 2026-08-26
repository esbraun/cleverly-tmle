"""Fast structural checks for the registered MSM evidence studies."""

from __future__ import annotations

import numpy as np
import pytest

from tests import discrete_law as point_law
from tests import discrete_law_longitudinal as longitudinal_law
from tests.studies import canonical_longitudinal_msm as longitudinal_study
from tests.studies import canonical_point_msm as point_study
from tests.studies import longitudinal_msm_properties as longitudinal_properties
from tests.studies import ltmle_properties
from tests.studies import point_msm_properties as point_properties


def test_point_projection_truth_and_curve_are_independent_finite_law_oracles() -> None:
    curves = point_properties.influence_curves()
    assert curves.shape == (len(point_law.SUPPORT), len(point_properties.TERMS))
    np.testing.assert_allclose(point_law.PROBS.reshape(-1) @ curves, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        list(point_properties.TRUTH.values()),
        point_properties.coefficients(point_law.PROBS),
        atol=1e-14,
    )


def test_point_projection_weights_are_load_bearing() -> None:
    declared = point_properties.coefficients(point_law.PROBS)
    uniform = point_properties.coefficients(point_law.PROBS, uniform=True)
    index = point_properties.TERMS.index("W")
    assert abs(declared[index] - uniform[index]) > 0.1


def test_longitudinal_projection_preserves_the_joint_regimen_curve() -> None:
    curves = longitudinal_properties.influence_curves()
    assert curves.shape == (len(longitudinal_law.SUPPORT), len(longitudinal_properties.TERMS))
    np.testing.assert_allclose(longitudinal_law.PROBS @ curves, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        list(longitudinal_properties.TRUTH.values()),
        longitudinal_properties.coefficients(longitudinal_law.PROBS),
        atol=1e-14,
    )


def test_longitudinal_projection_weights_are_load_bearing() -> None:
    declared = longitudinal_properties.coefficients(longitudinal_law.PROBS)
    uniform = longitudinal_properties.coefficients(
        longitudinal_law.PROBS,
        dict.fromkeys(longitudinal_properties.LABELS, 1.0),
    )
    index = longitudinal_properties.TERMS.index("duration")
    assert abs(declared[index] - uniform[index]) > 0.09


def test_longitudinal_untargeted_control_differs_only_by_fluctuation_on_exact_fit() -> None:
    frame = longitudinal_law.frame()
    result = longitudinal_properties.fit(frame, "both_correct")
    means = {
        label: ltmle_properties.untargeted(frame, label, "both_correct")
        for label in longitudinal_properties.LABELS
    }
    plug_in = longitudinal_properties.projection_operator() @ np.array(
        [means[label] for label in longitudinal_properties.LABELS]
    )
    targeted = np.array(
        [result[longitudinal_properties.NAMES[term]].psi for term in longitudinal_properties.TERMS]
    )
    np.testing.assert_allclose(plug_in, targeted, atol=1e-9)


def test_point_reference_uses_the_pinned_msm_parameter_and_custom_measure() -> None:
    source = (point_study.STUDY.artifacts / "run_study.R").read_text(encoding="utf-8")
    assert "Param_MSM$new" in source
    assert "tmle3_Spec_MSM$new" in source
    assert "1 + 0.5 * A + 5 * V" in source
    assert "class(weight)" in source


def test_longitudinal_reference_projects_separate_ltmle_fits_not_ltmle_msm() -> None:
    source = (longitudinal_study.STUDY.artifacts / "run_study.R").read_text(encoding="utf-8")
    assert "do.call(ltmle, arguments)" in source
    assert "operator %*%" in source
    assert "ltmleMSM" not in source


def test_the_longitudinal_sharp_null_has_zero_duration_coefficient() -> None:
    assert pytest.approx(0.0, abs=1e-14) == longitudinal_properties.NULL_TRUTH
