r"""Is the estimating equation's remainder second-order -- i.e. is the estimator doubly robust?

Double robustness is not something the targeting step manufactures by optimising.  It is a
property of the *estimating equation*: writing the von Mises expansion of the plug-in at
some pair of nuisance guesses :math:`(\hat g, \bar Q)`,

.. math::

    \Psi(\bar Q) - \Psi(P_0) + P_0 D^*(\hat g, \bar Q) = R_2(\hat g, \bar Q),

the estimator is consistent whenever :math:`R_2` vanishes.  What makes TMLE doubly robust
is that this remainder is a *product* of the two nuisance errors, so it is zero when either
factor is -- neither nuisance has to be right on its own.  For the counterfactual means and
their difference,

.. math::

    R_2 = \int \frac{\hat g - g_0}{\hat g}\,(\bar Q_1 - \bar Q_{0,1})\, dP_0
        + \int \frac{\hat g - g_0}{1 - \hat g}\,(\bar Q_0 - \bar Q_{0,0})\, dP_0 ,

and that identity is the whole content of the claim.

On the finite-support law of :mod:`tests.discrete_law` all three terms of the expansion are
exact finite sums, so this can be checked deterministically to machine precision rather
than inferred from a simulation.  :math:`\Psi` and :math:`R_2`'s closed form are written
out longhand here; :math:`D^*` is the library's -- which is the point, since the claim
under test is that the library's influence curve has the product remainder.

Two things this deliberately does *not* rest on.  It does not run the targeting step, so it
cannot be satisfied by a fluctuation that merely converged: the remainder is a property of
the estimating equation, and it is evaluated at nuisances that are *wrong on purpose*.  And
it does not share a derivation with :mod:`tests.unit.test_influence_gateaux`, which
establishes separately that :math:`D^*` is the efficient influence function; together the
two say the library solves the right equation and that equation is doubly robust.

Scope: the ``mean`` submodel, which is the one behind ``ey1``, ``ey0`` and ``ate``, and the
setting in which double robustness is classically stated.  The ``att`` and ``atc``
remainders have a different form and are not covered here; their influence curves are
checked in the Gateaux module.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import counterfactual_means
from tests import discrete_law as law

#: A propensity that is wrong at every covariate value, and stays inside ``(0, 1)``.
WRONG_G = np.array([0.55, 0.35, 0.45])

#: An outcome regression that is wrong in both arms, with mixed signs so that errors
#: cannot cancel into a spurious pass.
WRONG_Q = law.Q + np.array([[0.10, -0.15], [-0.20, 0.10], [0.05, 0.20]])


def _expansion(g_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    r"""``R_2`` for ``ey1``, ``ey0`` and ``ate`` at the given nuisance guesses.

    The plug-in and ``P_0 D^*`` both come from the library, evaluated at nuisances it did
    not fit and with no targeting step: ``InitialFit`` holds ``Qbar`` as handed to it, and
    the clever covariate is built from ``g_hat``.  Because the sample realises the law
    exactly, the sample mean of the influence curve *is* :math:`P_0 D^*`.
    """
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    treatment = frame["A"].to_numpy(dtype=float)
    outcome = frame["Y"].to_numpy(dtype=float)

    at_one, at_zero = q_hat[covariate, 1], q_hat[covariate, 0]
    initial = InitialFit(
        observed=np.where(treatment == 1.0, at_one, at_zero),
        at_one=at_one,
        at_zero=at_zero,
    )
    submodel = submodel_for("mean", treatment, g_hat[covariate])
    psi_one, ic_one, psi_zero, ic_zero = counterfactual_means(
        outcome, initial, submodel, np.ones(law.N)
    )

    remainder_one = psi_one - law.TRUTH["ey1"] + float(np.mean(ic_one))
    remainder_zero = psi_zero - law.TRUTH["ey0"] + float(np.mean(ic_zero))
    return {
        "ey1": remainder_one,
        "ey0": remainder_zero,
        "ate": remainder_one - remainder_zero,
    }


def _product_form(g_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    """The remainder as theory says it must be: a product of the two nuisance errors."""
    g_error = g_hat - law.G
    one = float(np.sum(law.P_W * g_error / g_hat * (q_hat[:, 1] - law.Q[:, 1])))
    zero = float(-np.sum(law.P_W * g_error / (1.0 - g_hat) * (q_hat[:, 0] - law.Q[:, 0])))
    return {"ey1": one, "ey0": zero, "ate": one - zero}


ESTIMANDS = ("ey1", "ey0", "ate")


class TestTheRemainderIsAProductOfNuisanceErrors:
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_matches_the_closed_form(self, name: str) -> None:
        # Both nuisances wrong, so every factor is active and nothing is zero by accident.
        actual = _expansion(WRONG_G, WRONG_Q)[name]
        assert actual == pytest.approx(_product_form(WRONG_G, WRONG_Q)[name], abs=1e-12)
        assert abs(actual) > 1e-3, "the misspecification is too mild to test anything"

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_vanishes_when_the_propensity_is_right(self, name: str) -> None:
        # Qbar arbitrarily wrong.  This is the half of double robustness that IPW-style
        # weighting supplies.
        assert _expansion(law.G, WRONG_Q)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_vanishes_when_the_outcome_regression_is_right(self, name: str) -> None:
        # g arbitrarily wrong.  This is the half the outcome model supplies.
        assert _expansion(WRONG_G, law.Q)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_vanishes_when_both_are_right(self, name: str) -> None:
        assert _expansion(law.G, law.Q)[name] == pytest.approx(0.0, abs=1e-12)


class TestTheRemainderIsSecondOrder:
    """Shrinking both nuisance errors by ``t`` has to shrink the remainder by ``t^2``.

    The closed form above already implies this -- a product of two ``O(t)`` factors -- but
    stating it as a rate is what connects the algebra to the condition the estimator
    actually needs: :math:`R_2 = o_P(n^{-1/2})` under a *product* rate on the two nuisance
    estimators, not a rate on either one alone.
    """

    @staticmethod
    def _at(t: float) -> float:
        g_hat = law.G + t * (WRONG_G - law.G)
        q_hat = law.Q + t * (WRONG_Q - law.Q)
        return _expansion(g_hat, q_hat)["ate"]

    def test_halving_the_nuisance_error_quarters_the_remainder(self) -> None:
        # Not exactly 4: the 1/g_hat factor moves with t too, so the ratio approaches 4
        # from below as t shrinks.  It reaches 3.991 by t = 0.005.
        assert self._at(0.005) / self._at(0.0025) == pytest.approx(4.0, abs=0.05)

    def test_the_remainder_is_negligible_beside_a_first_order_error(self) -> None:
        # At a 1% nuisance error the remainder is smaller by two orders of magnitude,
        # which is the practical content of "second-order".
        assert abs(self._at(0.01)) < 0.01 * abs(self._at(1.0))
