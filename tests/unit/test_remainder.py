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

Two things this deliberately does *not* rest on.  The remainder assertions do not run the
targeting step, so they cannot be satisfied by a fluctuation that merely converged: the
remainder is a property of the estimating equation, and it is evaluated at nuisances that
are *wrong on purpose*.  And nothing here shares a derivation with
:mod:`tests.unit.test_influence_gateaux`, which establishes separately that :math:`D^*` is
the efficient influence function; together the two say the library solves the right equation
and that equation is doubly robust.

:class:`TestTruncationRegularisesRatherThanRetargets` is the one exception to the first of
those, and for a reason: its closing claim is about *which* equation a truncated fit leaves
solved, which cannot be asked without solving one.

Scope: the ``mean`` submodel, which is the one behind ``ey1``, ``ey0`` and ``ate``, and the
setting in which double robustness is classically stated.  The ``att`` and ``atc``
remainders have a different form and are not covered here; their influence curves are
checked in the Gateaux module.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation._score import score_columns
from cleverly.fluctuation.iterative import InitialFit, solve_fluctuation
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
        arms={1.0: at_one, 0.0: at_zero},
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


def _plug_in(g_hat: np.ndarray, q_hat: np.ndarray) -> tuple[float, float]:
    """``(psi_one, psi_zero)`` before any targeting, at the given nuisance guesses.

    Separated from :func:`_expansion` so the truncation class can compare plug-ins without
    also comparing remainders -- the whole point there being that one moves and the other
    does not.
    """
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    treatment = frame["A"].to_numpy(dtype=float)
    outcome = frame["Y"].to_numpy(dtype=float)

    at_one, at_zero = q_hat[covariate, 1], q_hat[covariate, 0]
    initial = InitialFit(
        observed=np.where(treatment == 1.0, at_one, at_zero),
        arms={1.0: at_one, 0.0: at_zero},
    )
    submodel = submodel_for("mean", treatment, g_hat[covariate])
    psi_one, _, psi_zero, _ = counterfactual_means(outcome, initial, submodel, np.ones(law.N))
    return psi_one, psi_zero


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


class TestTruncationRegularisesRatherThanRetargets:
    r"""What bounding :math:`g` away from 0 and 1 does, and what it does not do.

    Propensity truncation is the mechanism users actually reach for, and it is the one most
    easily described wrongly in both directions.  It does **not** redefine the estimand: the
    plug-in :math:`\int \bar Q(a, w)\, dP_n(w)` contains no propensity at all, and
    :math:`\Psi(P_0)` is a functional of the law, not of the estimator's settings.  Nor is
    it free: what it moves is :math:`R_2`, by exactly the closed form above evaluated at the
    bounded value, so the cost is priced by the same product formula as any other
    misspecification.  Truncation buys variance and pays in second-order bias.

    The subtler point, and the one this class exists for, is *which* estimating equation
    the fit then solves.  It solves the one belonging to the **truncated** mechanism, to
    machine precision -- not an approximation to the equation belonging to the untruncated
    truth.  ``score_check()`` therefore passes on a heavily truncated fit and is right to,
    which is worth knowing before reading a passing score check as evidence that truncation
    did no harm.

    The counterpart for the missingness mechanism is
    :class:`tests.unit.test_remainder_mar.TestTruncationRegularisesRatherThanRetargets`, and
    for the intermediate mechanism
    :class:`tests.unit.test_remainder_cde.TestTruncationRegularisesRatherThanRetargets`.
    """

    #: Binds in all three cells: ``g = [0.40, 0.60, 0.25]`` becomes ``[0.45, 0.55, 0.45]``.
    BOUND = 0.45

    def _truncated(self) -> np.ndarray:
        bounded = np.clip(law.G, self.BOUND, 1.0 - self.BOUND)
        assert not np.allclose(bounded, law.G), "the bound has to actually bind"
        return bounded

    def test_the_plug_in_does_not_move(self) -> None:
        # Two fits differing only in the propensity, compared before any targeting: the
        # substitution estimator is an average of the outcome predictions over the sample's
        # covariates, and no propensity appears in it.
        untruncated = _plug_in(law.G, WRONG_Q)
        truncated = _plug_in(self._truncated(), WRONG_Q)
        assert truncated[0] == pytest.approx(untruncated[0], abs=1e-15)
        assert truncated[1] == pytest.approx(untruncated[1], abs=1e-15)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_what_moves_is_the_second_order_remainder(self, name: str) -> None:
        bounded = self._truncated()
        actual = _expansion(bounded, WRONG_Q)[name]
        assert actual == pytest.approx(_product_form(bounded, WRONG_Q)[name], abs=1e-12)
        assert abs(actual) > 1e-3, "the bound is not binding hard enough to test anything"

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_a_bound_that_does_not_bind_costs_nothing(self, name: str) -> None:
        # The `auto` bound at n = 1000 is 5 / (sqrt(n) log n) = 0.023 and the smallest
        # propensity on this law is 0.25, so the estimator runs on the unmodified
        # mechanism -- which is the premise the Gateaux module's exactness rests on.
        loose = np.clip(law.G, 0.023, 1.0 - 0.023)
        assert np.allclose(loose, law.G)
        assert _expansion(loose, WRONG_Q)[name] == pytest.approx(0.0, abs=1e-12)

    def test_the_solved_equation_is_the_truncated_one(self) -> None:
        """Targeting zeroes the truncated score, and leaves the untruncated one standing.

        Run the fluctuation with the bounded propensity, then evaluate the score of the
        resulting targeted fit twice: once against the clever covariate it was fitted with,
        once against the one built from the true propensity.  The first is zero to machine
        precision because that is the equation the solver was handed.  The second is not,
        and its size is the honest measure of what truncation cost -- invisible to any
        diagnostic that only ever recomputes the equation the estimator chose.
        """
        frame = law.frame()
        covariate = frame["W"].to_numpy().astype(int)
        treatment = frame["A"].to_numpy(dtype=float)
        outcome = frame["Y"].to_numpy(dtype=float)
        weights = np.ones(law.N)

        at_one, at_zero = WRONG_Q[covariate, 1], WRONG_Q[covariate, 0]
        initial = InitialFit(
            observed=np.where(treatment == 1.0, at_one, at_zero),
            arms={1.0: at_one, 0.0: at_zero},
        )
        bounded = submodel_for("mean", treatment, self._truncated()[covariate])
        exact = submodel_for("mean", treatment, law.G[covariate])

        fit = solve_fluctuation(outcome, initial, bounded, weights)
        mask = np.ones(law.N, dtype=bool)
        solved = score_columns(outcome, fit.targeted.observed, bounded.observed, weights, mask)
        unsolved = score_columns(outcome, fit.targeted.observed, exact.observed, weights, mask)

        # The score computed here is the one the solver reports, so this is the library's
        # own equation rather than a re-derivation of it.
        np.testing.assert_allclose(solved, fit.score, atol=1e-15, rtol=0)
        # The solver's tolerance is *relative* to the clever covariate's magnitude -- which
        # is why the floor here is 1e-9 rather than machine epsilon: at a bound of 0.45 the
        # covariate is around 2.2, so a relative 1e-10 is an absolute 1e-11.
        assert fit.converged
        assert np.max(np.abs(solved)) < 1e-9
        assert np.max(np.abs(unsolved)) > 1e-3
        # Eight orders of magnitude apart, so this is a difference in kind rather than a
        # tolerance being generous to one side.
        assert np.max(np.abs(unsolved)) > 1e5 * np.max(np.abs(solved))


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
