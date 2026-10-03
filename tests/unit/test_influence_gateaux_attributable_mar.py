r"""Exact-law evidence for PAR and PAF with outcomes missing at random.

A joint fit stacks two shipped parents on the same rows: the natural-course mean
:math:`\psi_{\mathrm{obs}}=E\{m(A,W)\}` and the reference-arm mean
:math:`\psi_0=E\{m(0,W)\}`, with :math:`m(A,W)=E(Y\mid\Delta=1,A,W)`.  Linearity gives the
PAR curve :math:`D_{\mathrm{obs}}-D_0`, and the delta method gives the PAF curve
:math:`-D_0/\psi_{\mathrm{obs}}+\psi_0D_{\mathrm{obs}}/\psi_{\mathrm{obs}}^2`.

:mod:`tests.discrete_law_mar` makes every conditional probability exact, so the reported
curves are compared with the definition-level Gateaux derivative of
:func:`tests.discrete_law_mar.functional` at each of its eighteen support points.  The
response-score terms are nonzero at the truth, so the pointwise comparison sees them
there.  The variance tests see the cross-covariance of the two parent curves, which is
large on this law (exact correlation 0.775).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.estimators import TMLE
from cleverly.inference.cluster import influence_variance
from tests import discrete_law_mar as law
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment

JOINT = ("ey_obs", "ey0", "par", "paf")


def _estimator(dgp: Any, estimands: tuple[str, ...] = JOINT) -> TMLE:
    return TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        missingness_learner=OracleMissingness(dgp),
        cross_fit=False,
        fluctuation="logistic",
        targeting="iterative",
        target_weights=False,
        estimands=estimands,
        simultaneous=False,
        random_state=0,
        max_iter=100,
        tol=1e-12,
    )


def _fit(dgp: Any = None, estimands: tuple[str, ...] = JOINT, **roles: Any) -> Any:
    frame = roles.pop("frame", law.frame())
    return (
        _estimator(law.DiscreteLaw() if dgp is None else dgp, estimands)
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta", **roles)
        .single()
    )


@pytest.fixture(scope="module")
def exact_fit() -> Any:
    return _fit()


def _at_support(result: Any, name: str) -> np.ndarray:
    return np.asarray(result[name].influence_curve, dtype=float)[law.first_row_of()]


def _exact_moment(left: str, right: str) -> float:
    """``E_0[D_left D_right]`` on the law, from the Gateaux derivatives alone."""
    return float(np.sum(law.PROBS.reshape(-1) * law.eif(left) * law.eif(right)))


class TestTheJointFitIsTheStack:
    @pytest.mark.parametrize("name", JOINT)
    def test_each_point_is_the_exact_functional(self, exact_fit: Any, name: str) -> None:
        assert exact_fit.psi(name) == pytest.approx(law.TRUTH[name], abs=1e-12)

    @pytest.mark.parametrize("name", JOINT)
    def test_each_curve_is_the_gateaux_derivative(self, exact_fit: Any, name: str) -> None:
        np.testing.assert_allclose(_at_support(exact_fit, name), law.eif(name), atol=1e-12, rtol=0)

    def test_the_truths_are_the_declared_values(self) -> None:
        assert law.TRUTH["ey_obs"] == pytest.approx(0.494, abs=1e-12)
        assert law.TRUTH["ey0"] == pytest.approx(0.380, abs=1e-12)
        assert law.TRUTH["par"] == pytest.approx(0.114, abs=1e-12)
        assert law.TRUTH["paf"] == pytest.approx(0.2307692307692308, abs=1e-12)

    def test_the_par_curve_is_the_difference_of_the_parent_curves(self, exact_fit: Any) -> None:
        observed = np.asarray(exact_fit["ey_obs"].influence_curve)
        reference = np.asarray(exact_fit["ey0"].influence_curve)
        np.testing.assert_array_equal(
            np.asarray(exact_fit["par"].influence_curve), observed - reference
        )

    def test_two_fluctuations_are_solved_and_both_need_no_step(self, exact_fit: Any) -> None:
        assert tuple(exact_fit.fluctuations) == ("natural_course", "mean")
        for fluctuation in exact_fit.fluctuations.values():
            assert np.max(np.abs(fluctuation.epsilon)) == pytest.approx(0.0, abs=1e-12)

    def test_every_estimate_declares_the_centered_rule_and_curve_status(
        self, exact_fit: Any
    ) -> None:
        for estimate in exact_fit.estimates.values():
            assert estimate.covariance_rule == "centered"
            assert estimate.inference == "influence_curve"


class TestTheResponseScoreWitnesses:
    """W1 and W2: the response terms are nonzero, and each equals its parent's term."""

    @staticmethod
    def _response_term(curve: np.ndarray, plug_in: np.ndarray) -> np.ndarray:
        return curve - plug_in

    def test_w1_the_natural_path_term_is_the_parent_term(self, exact_fit: Any) -> None:
        scalar = _fit(estimands=("ey_obs",))
        joint = _at_support(exact_fit, "ey_obs")
        alone = _at_support(scalar, "ey_obs")
        for point, (w, a, kind) in enumerate(law.SUPPORT):
            plug_in = law.Q[w, a] - law.TRUTH["ey_obs"]
            term = joint[point] - plug_in
            if kind == law.UNOBSERVED:
                assert term == pytest.approx(0.0, abs=1e-12)
                continue
            assert abs(term) > 0.1
            assert term == pytest.approx((kind - law.Q[w, a]) / law.PI[w, a], abs=1e-12)
            assert term == pytest.approx(alone[point] - plug_in, abs=1e-12)

    def test_w2_the_reference_path_term_is_the_parent_term(self, exact_fit: Any) -> None:
        arm_only = _fit(estimands=("ey0",))
        joint = _at_support(exact_fit, "ey0")
        alone = _at_support(arm_only, "ey0")
        for point, (w, a, kind) in enumerate(law.SUPPORT):
            plug_in = law.Q[w, 0] - law.TRUTH["ey0"]
            term = joint[point] - plug_in
            if kind == law.UNOBSERVED or a != 0:
                assert term == pytest.approx(0.0, abs=1e-12)
                continue
            expected = (kind - law.Q[w, 0]) / ((1.0 - law.G[w]) * law.PI[w, 0])
            assert abs(term) > 0.1
            assert term == pytest.approx(expected, abs=1e-12)
            assert term == pytest.approx(alone[point] - plug_in, abs=1e-12)

    def test_w3_the_parent_curves_are_strongly_correlated(self) -> None:
        covariance = _exact_moment("ey_obs", "ey0")
        correlation = covariance / np.sqrt(
            _exact_moment("ey_obs", "ey_obs") * _exact_moment("ey0", "ey0")
        )
        assert covariance == pytest.approx(0.63641, abs=1e-5)
        assert correlation == pytest.approx(0.775, abs=1e-3)
        assert correlation > 0.5


class TestTheVarianceReadsTheCrossCovariance:
    def test_the_par_variance_is_the_centered_variance_of_the_difference(
        self, exact_fit: Any
    ) -> None:
        observed = np.asarray(exact_fit["ey_obs"].influence_curve)
        reference = np.asarray(exact_fit["ey0"].influence_curve)
        n = observed.size
        expected = float(np.var(observed - reference, ddof=1) / n)
        assert exact_fit["par"].variance == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize(("name", "ratio"), [("par", 1.951), ("paf", 1.715)])
    def test_dropping_the_cross_covariance_moves_the_standard_error(
        self, exact_fit: Any, name: str, ratio: float
    ) -> None:
        """M1's witness: the parents' SEs added in quadrature are far too wide."""
        observed = np.asarray(exact_fit["ey_obs"].influence_curve)
        reference = np.asarray(exact_fit["ey0"].influence_curve)
        n = observed.size
        if name == "par":
            independent = np.var(observed, ddof=1) / n + np.var(reference, ddof=1) / n
        else:
            psi_obs, psi_0 = exact_fit.psi("ey_obs"), exact_fit.psi("ey0")
            independent = (np.var(reference, ddof=1) / psi_obs**2) / n + (
                np.var(observed, ddof=1) * psi_0**2 / psi_obs**4
            ) / n
        assert np.sqrt(independent / exact_fit[name].variance) == pytest.approx(ratio, abs=1e-3)

    def test_the_covariance_reads_the_same_row_curves(self, exact_fit: Any) -> None:
        observed = np.asarray(exact_fit["ey_obs"].influence_curve)
        reference = np.asarray(exact_fit["ey0"].influence_curve)
        n = observed.size
        matrix = np.asarray(exact_fit.covariance(["ey_obs", "ey0"]))
        expected = float(np.cov(observed, reference, ddof=1)[0, 1] / n)
        assert matrix[0, 1] == pytest.approx(expected, rel=1e-12)
        # The sample is the law, so the population moment is the same number up to n-1.
        assert matrix[0, 1] * (n - 1) == pytest.approx(_exact_moment("ey_obs", "ey0"), rel=1e-12)


#: Weight functions of the observed row, by label.  The first depends on ``W`` only, as
#: the study's weighted scenario does; the second also on the arm and the outcome.
WEIGHT_FUNCTIONS = {
    "baseline": lambda w, a, k: 0.5 + 0.25 * w,
    "arm_and_outcome": lambda w, a, k: 1.0 + 0.5 * a + 0.8 * (k == law.OBSERVED_ONE),
}


class TestFixedWeightsDefineATiltedLaw:
    @pytest.mark.parametrize("label", sorted(WEIGHT_FUNCTIONS))
    @pytest.mark.parametrize("name", JOINT)
    def test_the_weighted_curve_is_the_weighted_gateaux_derivative(
        self, label: str, name: str
    ) -> None:
        cells = law.cell_weights(WEIGHT_FUNCTIONS[label])
        tilted = law.DiscreteLaw(law.tilt(law.PROBS, cells))
        result = _fit(tilted, frame=law.frame().assign(w=law.row_weights(cells)), weights="w")
        assert result.psi(name) == pytest.approx(
            float(law.weighted_functional(law.PROBS, name, cells)), abs=1e-12
        )
        np.testing.assert_allclose(
            _at_support(result, name), law.weighted_eif(name, cells), atol=1e-11, rtol=0
        )


class TestClustersAreTheUnit:
    @pytest.mark.parametrize(
        ("clusters", "status"), [(50, "influence_curve"), (20, "few_cluster_plugin")]
    )
    def test_the_variance_sums_the_curve_within_clusters(self, clusters: int, status: str) -> None:
        frame = law.frame()
        codes = np.arange(len(frame)) % clusters
        result = _fit(frame=frame.assign(cluster=codes), id="cluster")
        curve = np.asarray(result["par"].influence_curve)
        assert result["par"].variance == pytest.approx(
            influence_variance(curve, cluster=codes), rel=1e-12
        )
        assert result["par"].variance != pytest.approx(influence_variance(curve), rel=1e-3)
        assert result.psi("par") == pytest.approx(law.TRUTH["par"], abs=1e-12)
        assert result["par"].inference == status
