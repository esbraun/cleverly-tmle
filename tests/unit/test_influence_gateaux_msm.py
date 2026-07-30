r"""Is the working model's influence curve the efficient influence function of ``beta``?

The same question :mod:`tests.unit.test_influence_gateaux` asks of the arm-indexed
estimands, asked of the projection

.. math::

    \beta(P) = M(P)^{-1}\, E_P\Big[\sum_a h(a, V)\,\varphi(a, V)\, \bar Q(a, W)\Big],
    \qquad
    M(P) = E_P\Big[\sum_a h(a, V)\,\varphi(a, V)\varphi(a, V)^\top\Big]

and answered the same way: differentiate a longhand statement of :math:`\beta` along the
contamination path by complex step, and compare against what the estimator reports.  No
clever covariate, no submodel and no library code enters the derivation.

Two things about this parameter make it easier to get wrong than the means, and both are
pinned below.  :math:`M` **depends on the law** -- through the marginal of :math:`V` -- so
treating it as a constant gives an influence curve that is wrong by exactly the term
:math:`M` contributes, and every check that holds :math:`P(W)` fixed would still pass.  And
the model is **not saturated** here: three coefficients against six ``(w, a)`` cells, so
``beta`` is a genuine projection and not a reparameterisation of the conditional means.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE
from cleverly.msm import MSM, MSMSet
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment

ESTIMANDS = tuple(law.PER_ARM_NAMES["msm"])


def working_model() -> MSM:
    """The library-side statement of the model :mod:`tests.discrete_law` declares.

    ``MSM.linear(modifiers=("W",), interaction=False)`` *is* ``[1, a, W]``, so the terms
    come from the shorthand rather than being written twice.  The weights are supplied
    explicitly because there is no shorthand for them, and because a uniform weight would
    make ``beta_a`` collapse to the ATE -- see :data:`law.MSM_WEIGHTS`.
    """
    return MSM.linear(
        modifiers=("W",),
        interaction=False,
        weights=lambda arm, frame: 1.0 + 0.5 * float(arm) + 0.25 * np.asarray(frame["W"]),
    )


@pytest.fixture(scope="module")
def exact_fit():
    """A working-model fit on the discrete law with oracle nuisances.

    Handed the truth, the initial fit is exactly right in the sample, ``epsilon`` is zero
    and the curve reported is the EIF at ``P_0`` rather than an estimate of it.
    """
    dgp = law.DiscreteLaw()
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        cross_fit=False,
        msm=working_model(),
        estimands="all",
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(law.frame(), outcome="Y", treatment="A").single()


class TestTheOracleAndTheLibraryDescribeTheSameModel:
    """The join between :data:`law.MSM_DESIGN` and :func:`working_model`.

    Everything below compares a library estimate against an oracle keyed by term name.  If
    the two sides disagreed about what ``msm[a]`` was the coefficient *of*, every later
    assertion would be comparing two different parameters and could pass while both were
    wrong.
    """

    def test_the_evaluated_design_and_weights_are_the_declared_ones(self) -> None:
        from cleverly.data import CausalData

        data = CausalData.from_frame(law.frame(), outcome="Y", treatment="A", covariates=["W"])
        evaluated = MSMSet.evaluate(working_model(), data)
        levels = np.rint(data.covariates[:, 0]).astype(int)
        np.testing.assert_array_equal(evaluated.design, law.MSM_DESIGN[levels])
        np.testing.assert_array_equal(evaluated.weights, law.MSM_WEIGHTS[levels])
        assert evaluated.terms == law.MSM_TERMS

    def test_the_model_is_not_saturated(self) -> None:
        """Three coefficients, six cells -- otherwise this checks nothing about ``M``."""
        assert len(law.MSM_TERMS) < law.MSM_DESIGN.shape[0] * law.MSM_DESIGN.shape[1]

    def test_the_weights_are_not_uniform(self) -> None:
        """A uniform weight would make ``beta_a`` the ATE identically; see the law."""
        assert law.MSM_WEIGHTS.min() < law.MSM_WEIGHTS.max()
        assert law.TRUTH["msm[a]"] != pytest.approx(law.TRUTH["ate"], abs=1e-6)


class TestThePremisesHold:
    def test_the_gateaux_derivative_has_mean_zero(self) -> None:
        # An influence function is centred by construction; a failure here would indict
        # the numerical derivative rather than the library.
        for name in ESTIMANDS:
            assert float((law.PROBS.reshape(-1) * law.eif(name)).sum()) == pytest.approx(
                0.0, abs=1e-12
            )

    def test_targeting_has_nothing_left_to_do(self, exact_fit) -> None:
        for fluctuation in exact_fit.fluctuations.values():
            assert np.max(np.abs(fluctuation.epsilon)) == pytest.approx(0.0, abs=1e-12)


class TestTheInfluenceCurveIsTheEIF:
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_matches_the_numerical_gateaux_derivative(self, exact_fit, name: str) -> None:
        reported = np.asarray(exact_fit.estimates[name].influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(reported, law.eif(name), atol=1e-12, rtol=0)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_point_estimate_is_the_functional(self, exact_fit, name: str) -> None:
        assert exact_fit.estimates[name].psi == pytest.approx(law.TRUTH[name], abs=1e-12)

    def test_the_score_equation_is_solved_exactly(self, exact_fit) -> None:
        """``P_n D_beta = 0``, which for this parameter holds for *two* separate reasons.

        The fluctuation zeroes the residual half; the weighted least squares zeroes the
        plug-in half by construction.  Checked at ``1e-13`` rather than through
        ``score_check``'s tolerance, because with oracle nuisances there is nothing left
        for either half to solve and the answer should be arithmetic-exact.
        """
        for name in ESTIMANDS:
            assert float(np.mean(exact_fit.estimates[name].influence_curve)) == pytest.approx(
                0.0, abs=1e-13
            )


class TestTheComparisonHasTeeth:
    """Deliberate-mutation controls: each plausible way of building this wrong.

    The window the assertions above use is ``1e-12``.  Each mutation here has to move the
    answer by more than ``1e-2`` -- four orders past it -- so that "the test passes"
    cannot mean "the test could not tell".
    """

    @staticmethod
    def _hand_written(*, gram_factor: float = 1.0, plugin: bool = True, uniform_h: bool = False):
        """``D_beta`` at the support points, written out longhand from the formula."""
        h = np.ones_like(law.MSM_WEIGHTS) if uniform_h else law.MSM_WEIGHTS
        p_w = law.PROBS.sum(axis=(1, 2))
        gram = gram_factor * np.einsum("wap,waq,wa,w->pq", law.MSM_DESIGN, law.MSM_DESIGN, h, p_w)
        moment = np.einsum("wap,wa,wa,w->p", law.MSM_DESIGN, h, law.Q_EXACT, p_w)
        beta = np.linalg.solve(gram, moment)

        rows = []
        for w, a, y in law.SUPPORT:
            g = law.G_EXACT[w] if a == 1 else 1.0 - law.G_EXACT[w]
            clever = h[w, a] * law.MSM_DESIGN[w, a] / g
            residual = clever * (y - law.Q_EXACT[w, a])
            if plugin:
                fitted = law.MSM_DESIGN[w] @ beta
                residual = residual + np.einsum(
                    "ap,a,a->p", law.MSM_DESIGN[w], h[w], law.Q_EXACT[w] - fitted
                )
            rows.append(np.linalg.solve(gram, residual))
        return np.array(rows)

    def test_the_longhand_curve_agrees_when_nothing_is_mutated(self) -> None:
        expected = np.column_stack([law.eif(name) for name in ESTIMANDS])
        np.testing.assert_allclose(self._hand_written(), expected, atol=1e-12, rtol=0)

    @pytest.mark.parametrize(
        ("kwargs", "why"),
        [
            ({"gram_factor": 1.05}, "M is off by a constant -- a missing normalisation"),
            ({"plugin": False}, "the plug-in half is missing, leaving only the score"),
            ({"uniform_h": True}, "the working model's weights were dropped"),
        ],
    )
    def test_a_mutation_moves_the_answer(self, kwargs, why: str) -> None:
        expected = np.column_stack([law.eif(name) for name in ESTIMANDS])
        assert np.max(np.abs(self._hand_written(**kwargs) - expected)) > 1e-2, why
