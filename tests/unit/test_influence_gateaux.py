r"""Is the influence curve the library reports actually *the* efficient influence function?

The score check (:mod:`cleverly.validation.score`) verifies that targeting drove
:math:`P_n \hat D^*` to zero.  That is a statement about convergence of the fluctuation,
not about :math:`\hat D^*` being the right function: a wrong clever covariate used
consistently in both the targeting step and the influence curve would solve its own
equation to machine precision and pass.  The comparison against an independently written
AIPW estimator in :mod:`tests.e2e.test_oracle` narrows the gap but does not close it,
because AIPW is built from the same claimed influence curve.

This module closes it, from the definition.  For a pathwise-differentiable parameter in a
nonparametric model the efficient influence function is the Gateaux derivative of the
parameter along the path that contaminates :math:`P_0` with a point mass,

.. math::

    D^*(o) = \left.\frac{d}{dt}\, \Psi\bigl((1 - t) P_0 + t\,\delta_o\bigr)\right|_{t=0},

so on a law with finite support -- see :mod:`tests.discrete_law` -- it can be computed
*numerically* from a longhand statement of :math:`\Psi`, with no clever covariate, no
submodel and no library code anywhere in the derivation.  Whatever the estimator reports
has to equal it.

Two properties of that law make the comparison exact rather than approximate.  The sample
realises the law exactly, so the initial fit handed the oracle nuisances is correct *in
the sample*, ``epsilon_hat`` is zero, and the reported influence curve is the EIF at
:math:`P_0` rather than an estimate of it.  And the derivative is taken by complex step,
which has no truncation or cancellation error at double precision.  So the assertions
below are at ``1e-12``, and would fail deterministically -- not on an unlucky seed -- if a
clever covariate were wrong by any amount that mattered.

Every estimand is covered, which matters most for ``att`` and ``atc``: their influence
curves carry a term beyond "clever covariate times residual" because the estimand
conditions on the random event ``A = 1``, and dropping it is a known, silent bug that the
score check cannot see.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment

ESTIMANDS = ("ey1", "ey0", "ate", "att", "atc", "rr", "or")


@pytest.fixture(scope="module")
def exact_fit():
    """TMLE on the discrete law with the oracle nuisances, so ``P_n`` and ``Qbar`` are exact.

    ``cross_fit=False`` because there is nothing to cross-fit: the oracle does not learn
    from the data, and out-of-fold prediction would only add fold bookkeeping to a fit
    whose answer is already determined.
    """
    dgp = law.DiscreteLaw()
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        cross_fit=False,
        estimands="all",
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(law.frame(), outcome="Y", treatment="A")


class TestTheSampleRealisesTheLaw:
    """The premises the rest of the module rests on, asserted rather than assumed."""

    def test_the_empirical_conditional_law_is_the_true_one(self) -> None:
        frame = law.frame()
        assert len(frame) == law.N
        for w in range(3):
            rows = frame["W"] == w
            assert frame.loc[rows, "A"].mean() == pytest.approx(law.G[w], abs=1e-15)
            for a in range(2):
                arm = rows & (frame["A"] == a)
                assert frame.loc[arm, "Y"].mean() == pytest.approx(law.Q[w, a], abs=1e-15)

    def test_the_gateaux_derivative_has_mean_zero(self) -> None:
        # An influence function is centred by construction.  If this failed, the
        # numerical derivative -- not the library -- would be the thing that is wrong.
        for name in ESTIMANDS:
            assert float((law.PROBS.reshape(-1) * law.eif(name)).sum()) == pytest.approx(
                0.0, abs=1e-12
            )

    def test_targeting_has_nothing_left_to_do(self, exact_fit) -> None:
        # The initial fit is exactly correct in the sample, so the fluctuation's score is
        # already zero at epsilon = 0 and epsilon_hat is exactly zero.  This is what makes
        # the reported influence curve the EIF at P_0 rather than an estimate of it.
        for fluctuation in exact_fit.fluctuations.values():
            assert np.max(np.abs(fluctuation.epsilon)) == pytest.approx(0.0, abs=1e-12)


class TestTheInfluenceCurveIsTheEIF:
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_matches_the_numerical_gateaux_derivative(self, exact_fit, name: str) -> None:
        estimate = exact_fit.estimates[name]
        reported = np.asarray(estimate.influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(reported, law.eif(name), atol=1e-12, rtol=0)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_point_estimate_is_the_functional(self, exact_fit, name: str) -> None:
        # P_n is P_0 and the fluctuation is null, so the plug-in is the target parameter
        # itself.  Ratios are compared where their influence curve lives, on the log scale.
        estimate = exact_fit.estimates[name]
        psi = estimate.log_psi if estimate.scale == "ratio" else estimate.psi
        assert psi == pytest.approx(law.TRUTH[name], abs=1e-12)

    def test_a_wrong_clever_covariate_would_be_caught(self) -> None:
        """The negative control: the comparison has teeth.

        Written out at the support points, the EIF of ``EY1`` is
        ``1{a=1}/g(w) * (y - Qbar(1, w)) + Qbar(1, w) - psi``.  Scaling that weight by
        1.05 -- the kind of error targeting and inference would *share*, and which the
        score check therefore cannot see -- has to move it well outside the ``1e-12``
        window the assertions above use.
        """

        def hand_written(weight: float) -> np.ndarray:
            values = []
            for w, a, y in law.SUPPORT:
                clever = weight * (a == 1) / law.G[w]
                values.append(clever * (y - law.Q[w, 1]) + law.Q[w, 1] - law.TRUTH["ey1"])
            return np.array(values)

        np.testing.assert_allclose(hand_written(1.0), law.eif("ey1"), atol=1e-12, rtol=0)
        assert np.max(np.abs(hand_written(1.05) - law.eif("ey1"))) > 1e-2
