"""Contrasts of several estimands, and the joint covariance they need.

``delta_method`` and ``influence_covariance`` were both written, both exported, and
both reachable from nothing.  These tests pin the wiring, and the one that carries
the weight is :meth:`TestAgainstTheClosedForm.test_difference_reproduces_the_ate`:
the ATE has a closed-form influence curve (``IC1 - IC0``) that is checked against
the discrete-law oracle elsewhere, so rebuilding it through the delta method is a
genuine cross-check of the delta method rather than a restatement.

It also pins a limit worth knowing: the default numerical gradient is accurate to
about 1e-10 relative, so a contrast is *not* a drop-in replacement for a closed-form
influence curve at the tolerance the Gateaux tests use.  Supplying the analytic
gradient makes it exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE
from tests import discrete_law as law


@pytest.fixture(scope="module")
def result():
    """A fit on the exactly-representable discrete law, with oracle-friendly settings."""
    return (
        TMLE(
            outcome_learner="glm",
            treatment_learner="glm",
            cross_fit=False,
            estimands="all",
            random_state=0,
        )
        .fit(law.frame(), outcome="Y", treatment="A", covariates=["W"])
        .single()
    )


def difference(p):  # type: ignore[no-untyped-def]
    return p[0] - p[1]


class TestAgainstTheClosedForm:
    def test_difference_reproduces_the_ate(self, result) -> None:  # type: ignore[no-untyped-def]
        """A numerically differentiated contrast against a hand-derived one."""
        contrast = result.contrast(difference, ["ey1", "ey0"])
        ate = result["ate"]
        assert contrast.psi == pytest.approx(ate.psi, abs=1e-14)
        assert contrast.variance == pytest.approx(ate.variance, rel=1e-9)
        # The central-difference gradient is the only source of error here.
        np.testing.assert_allclose(contrast.influence_curve, ate.influence_curve, rtol=0, atol=1e-9)

    def test_an_analytic_gradient_is_exact(self, result) -> None:  # type: ignore[no-untyped-def]
        """With the gradient supplied there is no numerical error left at all."""
        contrast = result.contrast(
            difference, ["ey1", "ey0"], gradient=lambda p: np.array([1.0, -1.0])
        )
        np.testing.assert_array_equal(contrast.influence_curve, result["ate"].influence_curve)

    def test_ratio_reproduces_the_risk_ratio(self, result) -> None:  # type: ignore[no-untyped-def]
        contrast = result.contrast(lambda p: p[0] / p[1], ["ey1", "ey0"])
        assert contrast.psi == pytest.approx(result["rr"].psi, rel=1e-12)

    def test_the_numerical_gradient_is_not_good_enough_for_1e_12(self, result) -> None:  # type: ignore[no-untyped-def]
        """The documented limit, asserted so the docstring cannot quietly rot.

        If this ever starts passing at 1e-14, the default gradient improved and the
        docstring on ``contrast`` should say so.
        """
        contrast = result.contrast(difference, ["ey1", "ey0"])
        error = np.max(np.abs(contrast.influence_curve - result["ate"].influence_curve))
        assert 0.0 < error < 1e-8


class TestCovariance:
    def test_reconstructs_a_known_variance(self, result) -> None:  # type: ignore[no-untyped-def]
        """var(ate) = var(ey1) + var(ey0) - 2 cov, from the joint matrix."""
        cov = result.covariance(["ey1", "ey0"])
        assert cov.shape == (2, 2)
        rebuilt = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
        assert rebuilt == pytest.approx(result["ate"].variance, rel=1e-12)

    def test_diagonal_matches_the_individual_variances(self, result) -> None:  # type: ignore[no-untyped-def]
        cov = result.covariance(["ey1", "ey0"])
        assert cov[0, 0] == pytest.approx(result["ey1"].variance, rel=1e-12)
        assert cov[1, 1] == pytest.approx(result["ey0"].variance, rel=1e-12)

    def test_is_symmetric(self, result) -> None:  # type: ignore[no-untyped-def]
        # ``atol=0`` with no ``rtol`` was numpy's default 1e-7 relative -- five orders
        # looser than anything this can plausibly be wrong by, on a quantity built from
        # one Gram product.  Not asserted exact: ``np.cov`` reaches a threaded gemm, and
        # whether C[i, j] and C[j, i] accumulate in the same order is a property of the
        # BLAS rather than of this package.  It is bit-symmetric on the BLAS here.
        cov = result.covariance()
        np.testing.assert_allclose(cov, cov.T, rtol=1e-12, atol=0)

    def test_ignoring_the_covariance_would_be_wrong(self, result) -> None:  # type: ignore[no-untyped-def]
        """The estimands are correlated; treating them as independent inflates var.

        This is the reason ``contrast`` exists rather than leaving users to combine
        two standard errors.
        """
        naive = result["ey1"].variance + result["ey0"].variance
        assert naive != pytest.approx(result["ate"].variance, rel=1e-3)

    def test_defaults_to_every_estimand(self, result) -> None:  # type: ignore[no-untyped-def]
        assert result.covariance().shape == (len(result.estimates), len(result.estimates))


class TestErrors:
    def test_unknown_estimand_is_refused(self, result) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(KeyError, match="not requested"):
            result.contrast(difference, ["ey1", "nope"])

    def test_empty_selection_is_refused(self, result) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="no estimands"):
            result.covariance([])


class TestClustering:
    def test_the_contrast_uses_the_cluster_level_variance(self) -> None:
        """A contrast must inherit the independent unit, not re-derive it from rows."""
        from cleverly.datasets import GENERATORS

        frame, _ = GENERATORS["clustered"](n=400, seed=9)
        covariates = [c for c in frame.columns if c.startswith("W")]
        result = (
            TMLE(outcome_learner="glm", treatment_learner="glm", n_folds=4, random_state=7)
            .fit(frame, outcome="Y", treatment="A", covariates=covariates, id="cluster")
            .single()
        )
        contrast = result.contrast(
            difference, ["ey1", "ey0"], gradient=lambda p: np.array([1.0, -1.0])
        )
        assert contrast.n_clusters == result["ate"].n_clusters < result.n
        assert contrast.variance == pytest.approx(result["ate"].variance, rel=1e-12)
