r"""Exact second-order remainder for the MAR natural-course mean.

For a candidate response score ``pi`` and targeted outcome regression ``m``, the
repository uses the expansion convention

.. math::

    \Psi(P)-\Psi(P_0)=-P_0D(P)+R_2(P,P_0),

with

.. math::

    R_2(P,P_0)=E_0[(1-\pi_0/\pi)(m-m_0)].

Every expectation below is an exact empirical sum over
:mod:`tests.discrete_law_mar`.  The library supplies the targeted regression; the right
side is evaluated independently from the law's arrays.  Thus the test catches the
remainder sign, either missing nuisance factor, and a score equation that targets a
different parameter.

What this file does **not** check is the *reported* influence curve.  Targeting solves
:math:`P_n[\Delta(Y-m^*)/\pi]=0` and :math:`\psi` is the plug-in average of
:math:`m^*`, so :math:`P_n D` is identically zero whatever curve the library reports,
and the expansion below reduces to :math:`\psi-\psi_0`.  Deleting the response-residual
term from the reported curve leaves every assertion here green.
:mod:`tests.unit.test_influence_gateaux_natural_course_mar` is what pins the curve, by
comparing it against the definition-level Gateaux derivative and against an independent
longhand construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.estimators import TMLE
from tests import discrete_law_mar as law
from tests.conftest import OracleMissingness, OracleOutcome
from tests.studies.missing_outcome_study_helpers import FailTreatment

WRONG_PI = law.PI + np.array([[0.30, -0.15], [-0.20, 0.25], [-0.35, 0.10]])
WRONG_Q = law.Q + np.array([[0.10, -0.15], [-0.15, 0.10], [0.05, 0.20]])


def _fit(pi_hat: np.ndarray, q_hat: np.ndarray) -> Any:
    candidate = law.DiscreteLaw()
    candidate.pi = np.asarray(pi_hat, dtype=float)
    candidate.q = np.asarray(q_hat, dtype=float)
    return (
        TMLE(
            outcome_learner=OracleOutcome(candidate),
            treatment_learner=FailTreatment(),
            missingness_learner=OracleMissingness(candidate),
            cross_fit=False,
            fluctuation="logistic",
            targeting="iterative",
            target_weights=False,
            estimands=("ey_obs",),
            simultaneous=False,
            random_state=0,
            max_iter=100,
            tol=1e-12,
        )
        .fit(law.frame(), outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )


def _fluctuation(result: Any) -> Any:
    assert len(result.fluctuations) == 1
    return next(iter(result.fluctuations.values()))


def _expansion(result: Any) -> float:
    estimate = result["ey_obs"]
    return float(estimate.psi - law.TRUTH["ey_obs"] + np.mean(np.asarray(estimate.influence_curve)))


def _exact_remainder(result: Any, pi_hat: np.ndarray) -> float:
    frame = law.frame()
    w = frame["W"].to_numpy(dtype=int)
    a = frame["A"].to_numpy(dtype=int)
    m_star = np.asarray(_fluctuation(result).targeted.observed, dtype=float)
    m_zero = law.Q[w, a]
    return float(np.mean((1.0 - law.PI[w, a] / pi_hat[w, a]) * (m_star - m_zero)))


class TestTheExactRemainder:
    def test_both_wrong_matches_the_nonzero_signed_formula(self) -> None:
        result = _fit(WRONG_PI, WRONG_Q)
        actual = _expansion(result)
        expected = _exact_remainder(result, WRONG_PI)
        assert actual == pytest.approx(expected, abs=1e-12)
        assert actual == pytest.approx(0.0470893082419392, abs=1e-12)
        assert actual != pytest.approx(-expected, abs=1e-3)

    def test_a_correct_outcome_regression_rescues_a_wrong_response_score(self) -> None:
        result = _fit(WRONG_PI, law.Q)
        assert _expansion(result) == pytest.approx(0.0, abs=1e-12)
        assert _exact_remainder(result, WRONG_PI) == pytest.approx(0.0, abs=1e-12)

    def test_a_correct_response_score_rescues_a_wrong_outcome_regression(self) -> None:
        result = _fit(law.PI, WRONG_Q)
        assert _expansion(result) == pytest.approx(0.0, abs=1e-12)
        assert _exact_remainder(result, law.PI) == pytest.approx(0.0, abs=1e-12)

    def test_both_correct_has_zero_remainder(self) -> None:
        result = _fit(law.PI, law.Q)
        assert _expansion(result) == pytest.approx(0.0, abs=1e-12)
        assert _exact_remainder(result, law.PI) == pytest.approx(0.0, abs=1e-12)


class TestTheRemainderIsSecondOrder:
    @staticmethod
    def _at(t: float) -> float:
        pi_hat = law.PI + t * (WRONG_PI - law.PI)
        q_hat = law.Q + t * (WRONG_Q - law.Q)
        result = _fit(pi_hat, q_hat)
        expansion = _expansion(result)
        assert expansion == pytest.approx(_exact_remainder(result, pi_hat), abs=1e-12)
        return expansion

    def test_halving_both_nuisance_errors_quarters_the_remainder(self) -> None:
        larger = self._at(0.005)
        smaller = self._at(0.0025)
        assert larger / smaller == pytest.approx(4.0, abs=0.02)

    def test_the_remainder_is_nonzero_before_the_quadratic_limit(self) -> None:
        assert abs(self._at(1.0)) > 1e-2
