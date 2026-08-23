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
doubly robust on exactly the terms the arm-indexed means are.

That closed form is the **identity link's**, and the exactness of it is the linearity of
:math:`U` in :math:`\beta` rather than anything about double robustness; under a link the
same expansion leaves a further term quadratic in :math:`\hat\beta - \beta_0`, so the
remainder is second-order without being zero.
:class:`TestUnderALinkTheRemainderIsSecondOrderRatherThanZero` measures the rate instead
of asserting the equality, and says why.

Two things fall out of the identity-link form that are worth stating in their own right.  The working model enters only as the
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
from cleverly.msm import MSMSet, solve_projection
from cleverly.utils.bounds import OutcomeScaler
from tests import discrete_law as law

#: The same deliberately wrong nuisances :mod:`tests.unit.test_remainder` uses, so the
#: modules disagree about nothing except which functional they expand.
WRONG_G = np.array([0.55, 0.35, 0.45])
WRONG_Q = law.Q + np.array([[0.10, -0.15], [-0.20, 0.10], [0.05, 0.20]])

TERMS = law.MSM_TERMS


def _expansion(g_hat: np.ndarray, q_hat: np.ndarray, link: str = "identity") -> dict[str, float]:
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
    model = MSMSet(law.MSM_TERMS, design, weights, (0.0, 1.0), link)  # type: ignore[arg-type]
    # The covariate is evaluated at the beta the plug-in lands on, which is the beta the
    # curve is taken at -- under a link those are the same solve and must be the same
    # number, which is what the alternation exists to achieve on a real fit.
    beta = solve_projection(design, weights, q_hat[covariate], np.ones(law.N), link).beta
    submodel = submodel_for("msm", treatment, g_hat[covariate], msm=model.weighted_design_at(beta))
    coefficients = msm_coefficients(
        outcome,
        initial,
        submodel,
        design,
        weights,
        np.ones(law.N),
        OutcomeScaler.identity(),
        link=link,
    )
    # The sample realises the law exactly, so the sample mean of the influence curve *is*
    # P_0 D*.
    return {
        term: coefficients[float(index)].psi
        - law.TRUTH[law.msm_names(link)[index]]
        + float(np.mean(coefficients[float(index)].influence_curve))
        for index, term in enumerate(TERMS)
    }


def _exact_remainder(g_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    """The remainder as theory says it must be: an exact signed sum carrying both errors."""
    mechanism = np.column_stack([1.0 - g_hat, g_hat])
    truth = np.column_stack([1.0 - law.G, law.G])
    factor = (mechanism - truth) / mechanism
    gram = np.einsum("wap,waq,wa,w->pq", law.MSM_DESIGN, law.MSM_DESIGN, law.MSM_WEIGHTS, law.P_W)
    moment = np.einsum(
        "wap,wa,wa,w->p", law.MSM_DESIGN, law.MSM_WEIGHTS, factor * (q_hat - law.Q), law.P_W
    )
    values = np.linalg.solve(gram, moment)
    return dict(zip(TERMS, (float(v) for v in values), strict=True))


class TestTheRemainderCarriesBothNuisanceErrors:
    @pytest.mark.parametrize("term", TERMS)
    def test_matches_the_closed_form(self, term: str) -> None:
        actual = _expansion(WRONG_G, WRONG_Q)[term]
        assert actual == pytest.approx(_exact_remainder(WRONG_G, WRONG_Q)[term], abs=1e-12)
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


class TestUnderALinkTheRemainderIsSecondOrderRatherThanZero:
    r"""What changes with a link, and it is exactly one thing.

    The expansion is the same statement,

    .. math::

        R_2 = (\hat\beta - \beta_0)
            + M^{-1} E_0\Big[\sum_a h\,\frac{dm}{d\beta}\,
              \frac{g_0}{\hat g}\,(\bar Q_0 - \bar Q)\Big],

    the second term being what is left of :math:`P_0 D^*` once the plug-in half is
    cancelled by the definition of :math:`\hat\beta`.  With the identity link :math:`U` is
    linear in :math:`\beta`, so :math:`\hat\beta - \beta_0` is *exactly*
    :math:`M^{-1}E[\sum_a h\varphi(\bar Q - \bar Q_0)]` and the two terms collapse into the
    remainder form above -- which is why the assertions there can be equalities.

    Under a link they collapse only to first order, and what is left is quadratic in
    :math:`\hat\beta - \beta_0`, hence second order in the outcome error.  So a *correct
    mechanism no longer drives the remainder to zero*, and asserting that it does would be
    asserting the identity link's algebra of a parameter that does not have it.  What is
    true, and what is checked here, is the rate: halve the errors and the remainder
    quarters.

    The other direction is untouched -- a correct outcome regression still gives exactly
    zero, for both links -- because then :math:`\hat\beta = \beta_0` and every factor of
    :math:`\bar Q_0 - \bar Q` is zero.  That is the stronger half of double robustness and
    it survives the link intact.
    """

    LINKS = ("log", "logit")

    @staticmethod
    def _worst(values: dict[str, float]) -> float:
        return max(abs(value) for value in values.values())

    @pytest.mark.parametrize("link", LINKS)
    def test_it_vanishes_when_the_outcome_regression_is_right(self, link: str) -> None:
        assert self._worst(_expansion(WRONG_G, law.Q, link)) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("link", LINKS)
    def test_it_vanishes_when_both_are_right(self, link: str) -> None:
        assert self._worst(_expansion(law.G, law.Q, link)) == pytest.approx(0.0, abs=1e-12)

    #: Successive halvings of the nuisance error.  The ratio of consecutive remainders is
    #: ``4`` for a second-order term, and it is approached rather than hit: at a *large*
    #: error the third-order terms are still visible, and they are what makes the coarsest
    #: ratio 3.5 rather than 4. So the statement checked is the limit -- the ratio must
    #: reach 4 as the error shrinks -- not a window at one perturbation size, which would
    #: be a claim about how big the third-order term happens to be.
    SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)

    @classmethod
    def _rates(cls, link: str, *, mechanism: bool) -> list[float]:
        """Ratios of successive remainders as the nuisance error is halved."""
        ratios: list[float] = []
        previous = None
        for scale in cls.SCALES:
            wrong_q = law.Q + scale * (WRONG_Q - law.Q)
            wrong_g = law.G + scale * (WRONG_G - law.G) if mechanism else law.G
            current = cls._worst(_expansion(wrong_g, wrong_q, link))
            assert current > 1e-12, "the perturbation is too small to measure a rate"
            if previous is not None:
                ratios.append(previous / current)
            previous = current
        return ratios

    @pytest.mark.parametrize("link", LINKS)
    def test_a_correct_mechanism_leaves_a_second_order_remainder(self, link: str) -> None:
        """Halve the outcome error, and what is left falls by four.

        With ``g`` correct the whole remainder is the quadratic term, so this is the
        cleanest measurement of it there is: no product term to contaminate the rate.
        """
        assert self._rates(link, mechanism=False)[-1] == pytest.approx(4.0, abs=0.25)

    @pytest.mark.parametrize("link", LINKS)
    def test_both_wrong_is_second_order_too(self, link: str) -> None:
        """The claim double robustness actually rests on, with neither nuisance right."""
        rates = self._rates(link, mechanism=True)
        assert rates[-1] == pytest.approx(4.0, abs=0.25), rates

    @pytest.mark.parametrize("link", LINKS)
    def test_it_is_not_first_order(self, link: str) -> None:
        """The negative control the rate needs: a first-order term would halve, not quarter.

        Without it "the ratio is near 4" could be read as a loose check that passed
        because the tolerance was wide; 2 is nowhere near the window at any scale.
        """
        assert min(self._rates(link, mechanism=True)) > 3.0

    @pytest.mark.parametrize("link", LINKS)
    def test_the_identity_link_s_exactness_is_not_quietly_assumed(self, link: str) -> None:
        """The negative control for the class above: here it really is not zero.

        If a future change made ``M`` or the plug-in half agree with the identity link's
        algebra, this remainder would collapse to zero and the rate checks would pass
        vacuously.
        """
        assert self._worst(_expansion(law.G, WRONG_Q, link)) > 1e-4
