r"""Exact second-order remainder of PAR and PAF with outcomes missing at random.

The repository's expansion convention is :math:`\Psi(\hat P)-\Psi(P_0)+P_0D(\hat P)=R`.
For the two parents of the joint fit,

.. math::

    R_{\mathrm{nc}} = E_0[(1-\pi_0/\hat\pi)(\hat m^*_{\mathrm{nc}}-m_0)](A,W),\qquad
    R_0 = E_0[(1-g_0\pi_0/(\hat g\hat\pi))(\hat Q^*-Q_0)](0,W),

where :math:`\hat m^*_{\mathrm{nc}}` is the natural-course targeted regression and
:math:`\hat Q^*` the ``mean`` group's.  Linearity gives the PAR remainder
:math:`R_{\mathrm{nc}}-R_0`, and the delta method gives the PAF identity in
:func:`_paf_identity`.  Every expectation is an exact sum over
:mod:`tests.discrete_law_mar`; the right sides read only the law's arrays and the fit's
targeted regressions.

The product-only case is the union-model boundary of the stack.  A correct product
:math:`\hat g\hat\pi` at the reference arm with a wrong :math:`\hat\pi` rescues the
reference arm and not the natural course, because the natural course divides by
:math:`\hat\pi` alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.estimators import TMLE
from tests import discrete_law_mar as law
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment

WRONG_PI = law.PI + np.array([[0.30, -0.15], [-0.20, 0.25], [-0.35, 0.10]])
WRONG_Q = law.Q + np.array([[0.10, -0.15], [-0.15, 0.10], [0.05, 0.20]])
WRONG_G = np.array([0.55, 0.35, 0.45])

#: A response mechanism wrong at every cell whose reference column exceeds the true
#: product ``g0(0 | w) pi0(0, w)``, so a treatment law that restores the product exists.
PRODUCT_PI = np.array([[0.50, 0.50], [0.80, 0.40], [0.95, 0.60]])
#: ``g`` chosen so that ``(1 - g) * PRODUCT_PI[:, 0]`` equals the true product at arm 0.
PRODUCT_G = 1.0 - (1.0 - law.G) * law.PI[:, 0] / PRODUCT_PI[:, 0]


def _fit(g_hat: np.ndarray, pi_hat: np.ndarray, q_hat: np.ndarray) -> Any:
    candidate = law.DiscreteLaw()
    candidate.g = np.asarray(g_hat, dtype=float)
    candidate.pi = np.asarray(pi_hat, dtype=float)
    candidate.q = np.asarray(q_hat, dtype=float)
    return (
        TMLE(
            outcome_learner=OracleOutcome(candidate),
            treatment_learner=OracleTreatment(candidate),
            missingness_learner=OracleMissingness(candidate),
            cross_fit=False,
            fluctuation="logistic",
            targeting="iterative",
            target_weights=False,
            estimands=("ey_obs", "ey0", "par", "paf"),
            simultaneous=False,
            random_state=0,
            max_iter=200,
            tol=1e-13,
        )
        .fit(law.frame(), outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )


def _rows() -> tuple[np.ndarray, np.ndarray]:
    frame = law.frame()
    return frame["W"].to_numpy(dtype=int), frame["A"].to_numpy(dtype=int)


def _remainders(result: Any, g_hat: np.ndarray, pi_hat: np.ndarray) -> tuple[float, float]:
    w, a = _rows()
    natural = np.asarray(result.fluctuations["natural_course"].targeted.observed, dtype=float)
    reference = np.asarray(result.fluctuations["mean"].targeted.arms[0.0], dtype=float)
    r_nc = float(np.mean((1.0 - law.PI[w, a] / pi_hat[w, a]) * (natural - law.Q[w, a])))
    product = (1.0 - law.G[w]) * law.PI[w, 0] / ((1.0 - g_hat[w]) * pi_hat[w, 0])
    r_0 = float(np.mean((1.0 - product) * (reference - law.Q[w, 0])))
    return r_nc, r_0


def _expansion(result: Any, name: str) -> float:
    estimate = result[name]
    curve = np.asarray(estimate.influence_curve, dtype=float)
    return float(estimate.psi - law.TRUTH[name] + np.mean(curve))


def _paf_identity(result: Any, r_nc: float, r_0: float) -> float:
    r"""The exact PAF expansion through the delta method.

    With :math:`f(x,y)=1-y/x`, substituting :math:`P_0\hat D_k=R_k-(\hat\psi_k-\psi_k)`
    into :math:`P_0\hat D_{\mathrm{PAF}}` leaves the two parent remainders and one
    product of parent errors:
    :math:`-R_0/\hat\psi_{\mathrm{obs}}+\hat\psi_0R_{\mathrm{nc}}/\hat\psi_{\mathrm{obs}}^2
    +(\hat\psi_{\mathrm{obs}}-\psi_{\mathrm{obs}})
    (\psi_0/\psi_{\mathrm{obs}}-\hat\psi_0/\hat\psi_{\mathrm{obs}})/\hat\psi_{\mathrm{obs}}`.
    """
    obs_hat, ref_hat = result.psi("ey_obs"), result.psi("ey0")
    obs, ref = law.TRUTH["ey_obs"], law.TRUTH["ey0"]
    return (
        -r_0 / obs_hat
        + ref_hat * r_nc / obs_hat**2
        + (obs_hat - obs) * (ref / obs - ref_hat / obs_hat) / obs_hat
    )


CASES = {
    "outcome_correct": (WRONG_G, WRONG_PI, law.Q),
    "outcome_and_response_wrong": (law.G, WRONG_PI, WRONG_Q),
    "product_only": (PRODUCT_G, PRODUCT_PI, WRONG_Q),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_par_remainder_is_the_difference_of_the_parent_remainders(case: str) -> None:
    g_hat, pi_hat, q_hat = CASES[case]
    result = _fit(g_hat, pi_hat, q_hat)
    r_nc, r_0 = _remainders(result, g_hat, pi_hat)
    assert _expansion(result, "par") == pytest.approx(r_nc - r_0, abs=1e-12)
    assert _expansion(result, "ey_obs") == pytest.approx(r_nc, abs=1e-12)
    assert _expansion(result, "ey0") == pytest.approx(r_0, abs=1e-12)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_paf_remainder_follows_the_delta_method(case: str) -> None:
    g_hat, pi_hat, q_hat = CASES[case]
    result = _fit(g_hat, pi_hat, q_hat)
    r_nc, r_0 = _remainders(result, g_hat, pi_hat)
    assert _expansion(result, "paf") == pytest.approx(_paf_identity(result, r_nc, r_0), abs=1e-12)


def test_a_correct_outcome_regression_zeroes_both_remainders() -> None:
    g_hat, pi_hat, q_hat = CASES["outcome_correct"]
    r_nc, r_0 = _remainders(_fit(g_hat, pi_hat, q_hat), g_hat, pi_hat)
    assert r_nc == pytest.approx(0.0, abs=1e-12)
    assert r_0 == pytest.approx(0.0, abs=1e-12)


def test_both_remainders_are_nonzero_when_the_outcome_and_response_are_wrong() -> None:
    g_hat, pi_hat, q_hat = CASES["outcome_and_response_wrong"]
    r_nc, r_0 = _remainders(_fit(g_hat, pi_hat, q_hat), g_hat, pi_hat)
    assert abs(r_nc) > 1e-2
    assert abs(r_0) > 1e-2
    assert abs(r_nc - r_0) > 1e-3


def test_a_correct_product_rescues_the_reference_arm_and_not_the_natural_course() -> None:
    """The witness for the union model's intersection: the product-only case."""
    np.testing.assert_allclose(
        (1.0 - PRODUCT_G) * PRODUCT_PI[:, 0], (1.0 - law.G) * law.PI[:, 0], rtol=0, atol=1e-15
    )
    assert np.min(np.abs(PRODUCT_PI - law.PI)) > 0.04
    g_hat, pi_hat, q_hat = CASES["product_only"]
    result = _fit(g_hat, pi_hat, q_hat)
    r_nc, r_0 = _remainders(result, g_hat, pi_hat)
    assert r_0 == pytest.approx(0.0, abs=1e-12)
    assert abs(r_nc) > 1e-2
    assert abs(result.psi("par") - law.TRUTH["par"]) > 1e-2
