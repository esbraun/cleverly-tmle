"""Fast structural checks for the registered MSM evidence studies."""

from __future__ import annotations

import numpy as np
import pandas as pd
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


def test_the_declared_measure_is_the_one_the_truth_contracts_over() -> None:
    """The fit's weights come off :data:`PROJECTION_WEIGHTS`, which :func:`truth` uses too.

    A projection coefficient is defined by its measure, so a study that declares one measure
    for its truth and fits under another reports a bias against a parameter it never
    estimated -- and every gate would still read green, because both halves are internally
    consistent.  This evaluates the fit's own closure over all six cells of the law's support.
    """
    frame = pd.DataFrame({"W": [0.0, 1.0, 2.0]})
    msm = point_study.declared_msm()
    fitted = np.column_stack([msm.weights(arm, frame) for arm in (0, 1)])
    np.testing.assert_array_equal(fitted, point_study.PROJECTION_WEIGHTS)
    uniform = point_study.declared_msm(uniform=True)
    np.testing.assert_array_equal(uniform.weights(1, frame), np.ones(3))


def test_point_reference_uses_the_pinned_msm_parameter_and_custom_measure() -> None:
    """The R runner's weight literal, rebuilt from the array rather than retyped here.

    The measure crosses a language boundary, so the two sides cannot share a declaration.
    What they can share is a derivation: the coefficients come out of
    :data:`PROJECTION_WEIGHTS` and are formatted into the expression the pinned runner must
    contain, so moving the array without moving the runner fails here rather than in a paired
    comparison nobody would read as a measure mismatch.
    """
    weights = point_study.PROJECTION_WEIGHTS
    base = float(weights[0, 0])
    per_arm = float(weights[0, 1] - weights[0, 0])
    per_covariate = float(weights[1, 0] - weights[0, 0])
    expected = np.array(
        [[base + per_arm * a + per_covariate * w for a in range(2)] for w in range(3)]
    )
    np.testing.assert_array_equal(
        expected, weights, err_msg="the declared measure is no longer additive in a and W"
    )

    source = (point_study.STUDY.artifacts / "run_study.R").read_text(encoding="utf-8")
    assert "Param_MSM$new" in source
    assert "tmle3_Spec_MSM$new" in source
    assert f"{base:g} + {per_arm:g} * A + {per_covariate:g} * V" in source
    assert "class(weight)" in source


def test_longitudinal_reference_projects_separate_ltmle_fits_not_ltmle_msm() -> None:
    source = (longitudinal_study.STUDY.artifacts / "run_study.R").read_text(encoding="utf-8")
    assert "do.call(ltmle, arguments)" in source
    assert "operator %*%" in source
    assert "ltmleMSM" not in source


def test_the_longitudinal_sharp_null_has_zero_duration_coefficient() -> None:
    assert pytest.approx(0.0, abs=1e-14) == longitudinal_properties.NULL_TRUTH


def test_the_point_sharp_null_has_zero_treatment_coefficient() -> None:
    """The type-I cell tests ``beta_a = 0``, so the null law's coefficient must *be* zero.

    ``replicate_row`` reads ``rejected`` off the estimate's own p-value, which is a test
    against zero, while the row's ``truth`` column carries ``NULL_TRUTH``.  The two agree
    only while ``NULL_TRUTH`` is zero; a null law whose coefficient drifted off it would
    publish a size against the wrong hypothesis and still pass, because nothing else in the
    cell reads the truth.

    It is zero for a reason worth naming before somebody edits the law.  ``NULL_Q`` is exactly
    linear in ``W`` and constant in ``A``, so the declared working model ``1 + a + W`` fits it
    without error.  A weighted least-squares projection of a function the model reproduces
    exactly returns that function's coefficients whatever the measure is -- which is why the
    ``a``-dependent projection weights do not move it.  Give ``NULL_Q`` any curvature in ``W``
    and that stops being true: the fit is then a genuine projection, the two arms see
    different effective ``W`` distributions through the weights, and ``beta_a`` is nonzero
    even though ``Q`` still does not depend on the arm.
    """
    assert pytest.approx(0.0, abs=1e-14) == point_properties.NULL_TRUTH
    null_q = point_properties.NULL_Q
    np.testing.assert_array_equal(null_q[:, 0], null_q[:, 1])
    curvature = null_q[2, 0] - 2.0 * null_q[1, 0] + null_q[0, 0]
    assert curvature == pytest.approx(0.0, abs=1e-14), (
        "the sharp null is zero only while the working model reproduces NULL_Q exactly"
    )
