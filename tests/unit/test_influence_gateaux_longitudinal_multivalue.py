"""Three-arm longitudinal TMLE against an independent finite-support functional."""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.longitudinal import LTMLE

from .. import discrete_law_longitudinal as binary_law
from .. import discrete_law_longitudinal_multivalue as law


@pytest.fixture(scope="module")
def fit() -> object:
    return LTMLE(
        law.REGIMEN_SPEC,
        reference=law.REFERENCE,
        outcome_learner=binary_law.CellMeans(),
        pseudo_learner=binary_law.CellMeans(),
        treatment_learner=law.CellProbabilities(),
        n_folds=1,
        g_bounds=(1e-8, 1.0 - 1e-8),
        simultaneous=False,
    ).fit(
        law.frame(),
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
    )


def test_every_reported_parameter_has_an_oracle_and_no_more(fit: object) -> None:
    assert set(fit) == set(law.NAMES)  # type: ignore[call-overload]
    assert set(law.TRUTH) == set(law.NAMES)


@pytest.mark.parametrize("name", law.NAMES)
def test_point_estimate_is_the_three_arm_g_formula(fit: object, name: str) -> None:
    assert fit.psi(name) == pytest.approx(law.TRUTH[name], abs=1e-12)  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", law.NAMES)
def test_influence_curve_is_the_gateaux_derivative(fit: object, name: str) -> None:
    reported = fit.influence_curves[name][law.first_row_of()]  # type: ignore[attr-defined]
    np.testing.assert_allclose(reported, law.eif_at(law.PROBS, name), atol=2e-13, rtol=0)


def test_the_second_order_remainder_is_nonzero_and_quadratic() -> None:
    """A real perturbation checks the von Mises remainder, not only the derivative."""
    name = "ey_regimen[respond]"
    direction = np.zeros_like(law.PROBS)
    # A cell on the rule's assigned A1 arm moves both its history distribution and a
    # later conditional mean, making the product-of-errors remainder visible.
    direction[12] = 1.0
    direction -= law.PROBS

    def remainder(epsilon: float) -> float:
        moved = law.PROBS + epsilon * direction
        return float(
            law.functional(moved, name)
            - law.functional(law.PROBS, name)
            + np.dot(law.PROBS, law.eif_at(moved, name))
        )

    coarse = remainder(0.0002)
    fine = remainder(0.0001)
    assert abs(fine) > 1e-8
    assert coarse / fine == pytest.approx(4.0, rel=0.03)


def test_assigned_arm_probability_is_not_a_binary_complement(fit: object) -> None:
    """Mutation witness: arm 2 must select its own column, not ``1 - P(A=1)``."""
    high = fit.fits["high"]  # type: ignore[attr-defined]
    first = high.cumulative[:, 0]
    w = np.asarray(law.frame()["W"], dtype=int)
    expected = law.G1[w, 2]
    binary_complement = 1.0 - law.G1[w, 1]
    np.testing.assert_allclose(first, expected, atol=1e-15, rtol=0)
    assert np.max(np.abs(first - binary_complement)) >= 0.25
