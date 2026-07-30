r"""Is the *working model's* estimating equation's remainder second-order?

The counterpart of :mod:`tests.unit.test_remainder` for the projection
:math:`\beta(P) = M^{-1} E_W \sum_a h(a, V)\,\varphi(a, V)\,\bar Q(a, W)`.  Expanding the
plug-in at a pair of nuisance guesses :math:`(\hat g, \bar Q)`,

.. math::

    \beta(\bar Q) - \beta(P_0) + P_0 D_\beta^*(\hat g, \bar Q) = R_2 ,
    \qquad
    R_2 = M^{-1} \sum_w P(w) \sum_a h(a, w)\,\varphi(a, w)\,
          \frac{\hat g(a \mid w) - g_0(a \mid w)}{\hat g(a \mid w)}\,
          \bigl(\bar Q(a, w) - \bar Q_0(a, w)\bigr),

again a *product* of the two nuisance errors -- so a working model's coefficients are
doubly robust on exactly the terms the arm-indexed means are.  Two things fall out of the
form that are worth stating in their own right.  The working model enters only as the
weight :math:`M^{-1} h \varphi` on the product, so **it cannot rescue either factor**, and a
misspecified working model does not make the remainder first-order: :math:`\beta` is a
projection, so there is no "model is wrong" bias term for it to have.  And :math:`M`
carries no nuisance at all, which is why it appears here as a constant matrix rather than
as a third error to expand.

Nothing here runs the targeting step.  The remainder is a property of the estimating
equation, evaluated at nuisances that are wrong on purpose, so a fluctuation that merely
converged cannot satisfy these assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import msm_coefficients
from cleverly.utils.bounds import OutcomeScaler
from tests import discrete_law as law

#: The same deliberately wrong nuisances :mod:`tests.unit.test_remainder` uses, so the
#: modules disagree about nothing except which functional they expand.
WRONG_G = np.array([0.55, 0.35, 0.45])
WRONG_Q = law.Q + np.array([[0.10, -0.15], [-0.20, 0.10], [0.05, 0.20]])

TERMS = law.MSM_TERMS


def _expansion(g_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    """``R_2`` per coefficient, with the plug-in and ``P_0 D*`` both taken from the library."""
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    treatment = frame["A"].to_numpy(dtype=float)
    outcome = frame["Y"].to_numpy(dtype=float)

    at_one, at_zero = q_hat[covariate, 1], q_hat[covariate, 0]
    initial = InitialFit(
        observed=np.where(treatment == 1.0, at_one, at_zero),
        arms={1.0: at_one, 0.0: at_zero},
    )
    design, weights = law.MSM_DESIGN[covariate], law.MSM_WEIGHTS[covariate]
    submodel = submodel_for("msm", treatment, g_hat[covariate], msm=design * weights[:, :, None])
    coefficients = msm_coefficients(
        outcome,
        initial,
        submodel,
        design,
        weights,
        np.ones(law.N),
        OutcomeScaler.identity(),
    )
    # The sample realises the law exactly, so the sample mean of the influence curve *is*
    # P_0 D*.
    return {
        term: coefficients[float(index)].psi
        - law.TRUTH[f"msm[{term}]"]
        + float(np.mean(coefficients[float(index)].influence_curve))
        for index, term in enumerate(TERMS)
    }


def _product_form(g_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    """The remainder as theory says it must be: a product of the two nuisance errors."""
    mechanism = np.column_stack([1.0 - g_hat, g_hat])
    truth = np.column_stack([1.0 - law.G, law.G])
    factor = (mechanism - truth) / mechanism
    gram = np.einsum("wap,waq,wa,w->pq", law.MSM_DESIGN, law.MSM_DESIGN, law.MSM_WEIGHTS, law.P_W)
    moment = np.einsum(
        "wap,wa,wa,w->p", law.MSM_DESIGN, law.MSM_WEIGHTS, factor * (q_hat - law.Q), law.P_W
    )
    values = np.linalg.solve(gram, moment)
    return dict(zip(TERMS, (float(v) for v in values), strict=True))


class TestTheRemainderIsAProductOfNuisanceErrors:
    @pytest.mark.parametrize("term", TERMS)
    def test_matches_the_closed_form(self, term: str) -> None:
        actual = _expansion(WRONG_G, WRONG_Q)[term]
        assert actual == pytest.approx(_product_form(WRONG_G, WRONG_Q)[term], abs=1e-12)
        assert abs(actual) > 1e-3, "the misspecification is too mild to test anything"

    @pytest.mark.parametrize("term", TERMS)
    def test_vanishes_when_the_mechanism_is_right(self, term: str) -> None:
        assert _expansion(law.G, WRONG_Q)[term] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("term", TERMS)
    def test_vanishes_when_the_outcome_regression_is_right(self, term: str) -> None:
        assert _expansion(WRONG_G, law.Q)[term] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("term", TERMS)
    def test_vanishes_when_both_are_right(self, term: str) -> None:
        assert _expansion(law.G, law.Q)[term] == pytest.approx(0.0, abs=1e-12)


class TestMisspecifyingTheWorkingModelIsNotABiasTerm:
    def test_a_wrong_working_model_leaves_the_remainder_second_order(self) -> None:
        """``beta`` is a projection, so "the model is wrong" is not an error to expand.

        The check that makes that concrete rather than asserted: with the mechanism
        correct the remainder is zero to machine precision *whatever* the working model
        says, and the model here says something quite wrong -- a line in ``a`` and ``W``
        against conditional means that are neither.
        """
        fitted = np.einsum("wap,p->wa", law.MSM_DESIGN, _beta_at(law.Q))
        assert np.max(np.abs(fitted - law.Q)) > 0.1, "the working model is not wrong enough"
        for value in _expansion(law.G, WRONG_Q).values():
            assert value == pytest.approx(0.0, abs=1e-12)


def _beta_at(q: np.ndarray) -> np.ndarray:
    """``beta`` from conditional means ``q``, longhand -- for the misspecification check."""
    gram = np.einsum("wap,waq,wa,w->pq", law.MSM_DESIGN, law.MSM_DESIGN, law.MSM_WEIGHTS, law.P_W)
    moment = np.einsum("wap,wa,wa,w->p", law.MSM_DESIGN, law.MSM_WEIGHTS, q, law.P_W)
    return np.asarray(np.linalg.solve(gram, moment))
