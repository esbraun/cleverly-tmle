r"""Is the working model's influence curve the efficient influence function of ``beta``?

The same question :mod:`tests.unit.test_influence_gateaux` asks of the arm-indexed
estimands, asked of the projection

.. math::

    U(\beta) = E_P\Big[\sum_a h(a, V)\,\frac{dm}{d\eta}\,\varphi(a, V)\,
               \big(\bar Q(a, W) - m(a, V; \beta)\big)\Big] = 0

and answered the same way: differentiate a longhand statement of :math:`\beta` along the
contamination path by complex step, and compare against what the estimator reports.  No
clever covariate, no submodel and no library code enters the derivation.

Run under **all three links**.  The identity link is the closed form
:math:`\beta = M^{-1} E[\sum_a h\varphi\bar Q]`; the others solve the same equation by
Newton, and the oracle runs its own Newton -- a fixed number of steps with no convergence
test, so that the functional stays analytic and the complex step differentiates *through*
the solve.  See :data:`tests.discrete_law.MSM_NEWTON_STEPS`.

Three things about this parameter make it easier to get wrong than the means, and each is
pinned below.  :math:`M` **depends on the law** -- through the marginal of :math:`V` -- so
treating it as a constant gives an influence curve that is wrong by exactly the term
:math:`M` contributes, and every check that holds :math:`P(W)` fixed would still pass.
Under a link :math:`M` also carries a **curvature term** that vanishes wherever the working
model fits, so a saturated model cannot see it.  And the model is **not saturated** here:
three coefficients against six ``(w, a)`` cells, so ``beta`` is a genuine projection and not
a reparameterisation of the conditional means.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE
from cleverly.msm import MSM, MSMSet
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment

#: Every link, checked on the same law and the same design.  The estimator reports its
#: coefficients as ``msm[term]`` whatever the link -- a fit declares one -- so the oracle
#: name is the one that varies; :func:`law.msm_names` is the map.
LINKS = tuple(law.MSM_LINKS)

REPORTED = tuple(f"msm[{term}]" for term in law.MSM_TERMS)


def working_model(link: str = "identity") -> MSM:
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
        link=link,  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module", params=LINKS)
def exact_fit(request):
    """A working-model fit on the discrete law with oracle nuisances, one link per case.

    Handed the truth, the initial fit is exactly right in the sample, ``epsilon`` is zero
    and the curve reported is the EIF at ``P_0`` rather than an estimate of it.  Under a
    link that is also what makes the alternation exit at once: with nothing for the
    fluctuation to do, ``beta`` does not move either.
    """
    dgp = law.DiscreteLaw()
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        cross_fit=False,
        msm=working_model(request.param),
        estimands="all",
        simultaneous=False,
        random_state=0,
    )
    return request.param, estimator.fit(law.frame(), outcome="Y", treatment="A").single()


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

    @pytest.mark.parametrize("link", [name for name in LINKS if name != "identity"])
    def test_the_links_describe_different_parameters(self, link: str) -> None:
        """Otherwise a link would be checked against the identity's oracle by accident."""
        identity = np.array([law.TRUTH[name] for name in law.msm_names("identity")])
        linked = np.array([law.TRUTH[name] for name in law.msm_names(link)])
        assert np.max(np.abs(identity - linked)) > 0.1


class TestTheNewtonSolveHasConverged:
    """The oracle solves its own normal equations by a *fixed* number of Newton steps.

    That is what keeps :func:`law.functional` analytic -- a convergence test is a
    comparison, and a comparison cannot be differentiated by a complex step.  The price is
    that the step count has to be shown to be enough, for the derivative as well as for
    the value, which is what this asserts: run it twice as long and nothing moves.
    """

    @pytest.mark.parametrize("link", [name for name in LINKS if name != "identity"])
    def test_doubling_the_steps_changes_neither_the_value_nor_the_curve(
        self, link: str, monkeypatch
    ) -> None:
        names = law.msm_names(link)
        before = {name: (law.functional(law.PROBS, name), law.eif(name)) for name in names}
        monkeypatch.setattr(law, "MSM_NEWTON_STEPS", 2 * law.MSM_NEWTON_STEPS)
        for name in names:
            value, curve = before[name]
            assert law.functional(law.PROBS, name) == pytest.approx(value, abs=1e-14)
            np.testing.assert_allclose(law.eif(name), curve, atol=1e-12, rtol=0)


class TestThePremisesHold:
    def test_the_gateaux_derivative_has_mean_zero(self) -> None:
        # An influence function is centred by construction; a failure here would indict
        # the numerical derivative rather than the library.
        for link in LINKS:
            for name in law.msm_names(link):
                assert float((law.PROBS.reshape(-1) * law.eif(name)).sum()) == pytest.approx(
                    0.0, abs=1e-12
                )

    def test_targeting_has_nothing_left_to_do(self, exact_fit) -> None:
        _, fit = exact_fit
        for fluctuation in fit.fluctuations.values():
            assert np.max(np.abs(fluctuation.epsilon)) == pytest.approx(0.0, abs=1e-12)

    def test_the_alternation_exits_without_moving_beta(self, exact_fit) -> None:
        """With an exact initial fit there is nothing for either equation to solve."""
        link, fit = exact_fit
        projection = fit.fluctuations["msm"].projection
        if link == "identity":
            assert projection is None
            return
        assert projection.converged and projection.n_outer == 1
        assert projection.trace[0][2] == pytest.approx(0.0, abs=1e-14)


#: Absolute window on the curve, passed explicitly with ``rtol=0`` as every sibling module
#: does.  ``1e-12`` under the identity link, where both sides evaluate a closed form and
#: agree to the last bit or two.  Under a link both sides run their own Newton solve and
#: the curve reaches magnitude 20 here, so ``1e-12`` absolute would be asking for ``5e-14``
#: *relative* through two independent iterations -- past what double precision gives. The
#: measured gap is ``1.3e-12``; ``1e-10`` leaves eight orders between a pass and the
#: smallest mutation ``TestTheComparisonHasTeeth`` has to catch.
CURVE_TOLERANCE = {"identity": 1e-12, "log": 1e-10, "logit": 1e-10}


class TestTheInfluenceCurveIsTheEIF:
    @pytest.mark.parametrize("index", range(len(REPORTED)))
    def test_matches_the_numerical_gateaux_derivative(self, exact_fit, index: int) -> None:
        link, fit = exact_fit
        reported = np.asarray(fit.estimates[REPORTED[index]].influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(
            reported,
            law.eif(law.msm_names(link)[index]),
            atol=CURVE_TOLERANCE[link],
            rtol=0,
        )

    @pytest.mark.parametrize("index", range(len(REPORTED)))
    def test_the_point_estimate_is_the_functional(self, exact_fit, index: int) -> None:
        link, fit = exact_fit
        assert fit.estimates[REPORTED[index]].psi == pytest.approx(
            law.TRUTH[law.msm_names(link)[index]], abs=1e-12
        )

    def test_the_score_equation_is_solved_exactly(self, exact_fit) -> None:
        """``P_n D_beta = 0``, which for this parameter holds for *two* separate reasons.

        The fluctuation zeroes the residual half; the weighted least squares zeroes the
        plug-in half by construction.  Checked at ``1e-13`` rather than through
        ``score_check``'s tolerance, because with oracle nuisances there is nothing left
        for either half to solve and the answer should be arithmetic-exact.
        """
        _, fit = exact_fit
        for name in REPORTED:
            assert float(np.mean(fit.estimates[name].influence_curve)) == pytest.approx(
                0.0, abs=1e-13
            )


class TestTheComparisonHasTeeth:
    """Deliberate-mutation controls: each plausible way of building this wrong.

    The window the assertions above use is ``1e-12``.  Each mutation here has to move the
    answer by more than ``1e-2`` -- four orders past it -- so that "the test passes"
    cannot mean "the test could not tell".
    """

    @staticmethod
    def _hand_written(
        link: str = "identity",
        *,
        gram_factor: float = 1.0,
        plugin: bool = True,
        uniform_h: bool = False,
        curvature: bool = True,
    ):
        """``D_beta`` at the support points, written out longhand from the formula."""
        inverse, slope, second = law.MSM_LINKS[link]
        h = np.ones_like(law.MSM_WEIGHTS) if uniform_h else law.MSM_WEIGHTS
        p_w = law.PROBS.sum(axis=(1, 2))
        beta = np.zeros(len(law.MSM_TERMS))
        for _ in range(law.MSM_NEWTON_STEPS):
            m = inverse(np.einsum("wap,p->wa", law.MSM_DESIGN, beta))
            residual = law.Q_EXACT - m
            u = np.einsum("wap,wa,w->p", law.MSM_DESIGN, h * slope(m) * residual, p_w)
            jacobian = np.einsum(
                "wap,waq,wa,w->pq",
                law.MSM_DESIGN,
                law.MSM_DESIGN,
                h * (slope(m) ** 2 - residual * second(m)),
                p_w,
            )
            beta = beta + np.linalg.solve(jacobian, u)

        m = inverse(np.einsum("wap,p->wa", law.MSM_DESIGN, beta))
        residual = law.Q_EXACT - m
        # The matrix the curve is premultiplied by the inverse of. `curvature=False` is
        # the mutation that matters under a link: M built as the slope-weighted Gram
        # matrix, which is right only where the working model fits.
        second_term = residual * second(m) if curvature else np.zeros_like(residual)
        gram = gram_factor * np.einsum(
            "wap,waq,wa,w->pq",
            law.MSM_DESIGN,
            law.MSM_DESIGN,
            h * (slope(m) ** 2 - second_term),
            p_w,
        )

        rows = []
        for w, a, y in law.SUPPORT:
            g = law.G_EXACT[w] if a == 1 else 1.0 - law.G_EXACT[w]
            clever = h[w, a] * slope(m)[w, a] * law.MSM_DESIGN[w, a] / g
            contribution = clever * (y - law.Q_EXACT[w, a])
            if plugin:
                contribution = contribution + np.einsum(
                    "ap,a,a,a->p", law.MSM_DESIGN[w], h[w], slope(m)[w], residual[w]
                )
            rows.append(np.linalg.solve(gram, contribution))
        return np.array(rows)

    @pytest.mark.parametrize("link", LINKS)
    def test_the_longhand_curve_agrees_when_nothing_is_mutated(self, link: str) -> None:
        expected = np.column_stack([law.eif(name) for name in law.msm_names(link)])
        np.testing.assert_allclose(self._hand_written(link), expected, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("link", LINKS)
    @pytest.mark.parametrize(
        ("kwargs", "why"),
        [
            ({"gram_factor": 1.05}, "M is off by a constant -- a missing normalisation"),
            ({"plugin": False}, "the plug-in half is missing, leaving only the score"),
            ({"uniform_h": True}, "the working model's weights were dropped"),
        ],
    )
    def test_a_mutation_moves_the_answer(self, link: str, kwargs, why: str) -> None:
        expected = np.column_stack([law.eif(name) for name in law.msm_names(link)])
        assert np.max(np.abs(self._hand_written(link, **kwargs) - expected)) > 1e-2, why

    @pytest.mark.parametrize("link", [name for name in LINKS if name != "identity"])
    def test_dropping_the_curvature_term_moves_the_answer(self, link: str) -> None:
        """The mutation a saturated working model could not catch.

        ``M`` without ``-(Qbar - m) d2m/deta2`` is the matrix an author would write by
        generalising the identity link's Gram matrix one factor at a time. It is right
        wherever the model fits, and this one does not fit -- three coefficients against
        six cells -- so the curve moves by two orders more than the tolerance.
        """
        expected = np.column_stack([law.eif(name) for name in law.msm_names(link)])
        gram_only = self._hand_written(link, curvature=False)
        assert np.max(np.abs(gram_only - expected)) > 1e-2

    def test_the_identity_link_has_no_curvature_to_drop(self) -> None:
        """Which is why it needed no such term, and why the link path needs its own check."""
        np.testing.assert_array_equal(
            self._hand_written("identity", curvature=False), self._hand_written("identity")
        )
